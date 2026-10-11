"""
Security Layer (revised)
Input normalization, injection screening, PII detection/masking,
secret + canary output validation.

NOTE: regex screening is a cheap first filter, not a real defense against
prompt injection. Pair it with a classifier and least-privilege tools.
"""

import hashlib
import logging
import re
import secrets
import unicodedata
from typing import Optional
from langsmith import traceable

logger = logging.getLogger("security")


def _fingerprint(text: str) -> str:
    """Short hash so blocked attempts can be correlated without storing content."""
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class InputSanitizer:
    """Normalize user input and screen for common injection phrasings."""

    INJECTION_PATTERNS = [
        r"\bignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions|prompts?|rules)\b",
        r"\bforget\s+(all\s+)?(the\s+)?(previous|prior|above|your\s+instructions)\b",
        r"\bdisregard\s+(all\s+)?(the\s+)?(previous|prior|above|your)\s+(instructions|rules)\b",
        r"\bnew\s+instructions\s*:",
        r"\b(reveal|show|print|repeat|leak|output)\b.{0,30}\b(system\s+prompt|your\s+instructions|hidden\s+prompt)\b",
        r"\bsystem\s+prompt\s*(is|:|override)",
        r"-{3,}\s*end\s*(of)?\s*(the\s+)?(system\s+)?prompt",
        r"\bpretend\s+(that\s+)?you\s*(are|'re)\s+(not\s+an?\s+ai|unrestricted|unfiltered|DAN)\b",
        r"\bact\s+as\s+if\s+you\s+(have\s+no|are\s+not|were\s+not)\b",
        r"\bbypass\s+(all\s+)?(your\s+)?(restrictions|safety|filters|guardrails)\b",
        r"\byou\s+are\s+now\s+(DAN|jailbroken|unrestricted)\b",
    ]

    ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
    SEPARATORS = re.compile(r"[-_=*.|~`]+")  # used to defeat "ig---nore" style evasion

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.INJECTION_PATTERNS]

    def normalize(self, text: str) -> str:
        """Fold homoglyphs/fullwidth chars, drop invisible chars, escape template braces."""
        text = unicodedata.normalize("NFKC", text)
        text = self.ZERO_WIDTH.sub("", text)
        text = text.replace("{{", "{ {").replace("}}", "} }")
        return text.strip()

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Check (already normalized) text.
        Returns: (is_safe, rejection_reason)
        """
        if len(text) > self.max_chars:
            return False, f"Blocked: input exceeds {self.max_chars} characters"

        # Scan the text as-is AND with separator junk removed, so
        # "ig---nore previous instructions" can't slip through.
        variants = (text, self.SEPARATORS.sub("", text))
        for pattern in self.patterns:
            if any(pattern.search(v) for v in variants):
                return False, "Blocked: potential prompt injection detected"
        return True, None


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------
def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class PIIDetector:
    """
    Detect and mask PII. Works on input (before LLM) and output (before client).
    US-centric; consider Microsoft Presidio for names, addresses, intl formats.
    """

    PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        # Candidates only; confirmed with Luhn below.
        "credit_card": re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"),
        # Requires separators/parentheses/+ so bare 10-digit IDs don't match.
        "phone": re.compile(
            r"(?<![\w])(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)"
        ),
        # Uganda/Kenya/Tanzania/Rwanda: +256 772 123456, 0772 123 456, 0414-123-456
        "phone_ea": re.compile(
            r"(?<![\w])(?:\+?(?:256|254|255|250)|0)[\s-]?[1-9]\d{1,2}[\s-]?\d{3}[\s-]?\d{3}(?!\d)"
        ),
    }

    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "phone_ea": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
    }

    def _matches(self, pii_type: str, text: str) -> list[re.Match]:
        found = list(self.PATTERNS[pii_type].finditer(text))
        if pii_type == "credit_card":
            found = [m for m in found if luhn_valid(m.group())]
        return found

    def detect(self, text: str) -> dict[str, int]:
        """Return {pii_type: count}. Counts only, so PII never ends up in logs."""
        result = {}
        for pii_type in self.PATTERNS:
            n = len(self._matches(pii_type, text))
            if n:
                result[pii_type] = n
        return result

    def values(self, text: str) -> set[str]:
        """Every PII string found in text (used to build an allowlist from trusted docs)."""
        return {m.group() for t in self.PATTERNS for m in self._matches(t, text)}

    def mask(self, text: str, allow: frozenset[str] | set[str] = frozenset()) -> str:
        """Replace PII with fixed markers (irreversible). Values in `allow` are kept."""
        return self._replace(
            text, lambda t, i, m: m.group() if m.group() in allow else self.MASK_MAP[t]
        )

    def mask_reversible(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace PII with numbered placeholders like [EMAIL_1] so the LLM can still
        reason about them. Returns (masked_text, mapping). Keep the mapping
        server-side and call restore() on the model's response.
        """
        mapping: dict[str, str] = {}
        counters: dict[str, int] = {}

        def make(pii_type: str, _i: int, m: re.Match) -> str:
            original = m.group()
            for token, value in mapping.items():
                if value == original:
                    return token
            counters[pii_type] = counters.get(pii_type, 0) + 1
            token = f"[{pii_type.upper()}_{counters[pii_type]}]"
            mapping[token] = original
            return token

        return self._replace(text, make), mapping

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        for token, original in mapping.items():
            text = text.replace(token, original)
        return text

    def _replace(self, text: str, repl) -> str:
        # Order matters: card before phone so card digits aren't half-eaten.
        for pii_type in self.PATTERNS:  # dict order: card runs before phones
            matches = self._matches(pii_type, text)
            for i, m in reversed(list(enumerate(matches))):
                text = text[: m.start()] + repl(pii_type, i, m) + text[m.end():]
        return text


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class OutputValidator:
    """
    Validate LLM output before returning to the client:
    PII leakage, real-looking secrets, and system-prompt (canary) leakage.
    """

    SECRET_PATTERNS = [
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),                       # OpenAI/Anthropic-style keys
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                            # AWS access key id
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),                  # GitHub tokens
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),                # Slack tokens
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWT
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ]

    def __init__(self, canary: Optional[str] = None, mask_pii: bool = True):
        self.pii_detector = PIIDetector()
        self.canary = canary
        self.mask_pii = mask_pii

    def validate(self, output: str, trusted_context: str = "") -> tuple[str, list[str]]:
        """
        trusted_context: text you vetted (e.g. retrieved company docs). PII that appears
        in it, like the company's own phone number, is allowed through; any other PII
        the model produces is masked.
        Returns: (cleaned_output, list_of_warnings)
        """
        warnings = []

        # System prompt leak: block outright.
        if self.canary and self.canary in output:
            logger.warning("canary leaked fp=%s", _fingerprint(output))
            return "[Response blocked: internal content leak]", ["Canary token leaked"]

        # Secrets: redact in place rather than discarding a useful answer.
        for pattern in self.SECRET_PATTERNS:
            if pattern.search(output):
                output = pattern.sub("[SECRET REDACTED]", output)
                warnings.append("Secret-format string redacted in output")

        # PII
        if self.mask_pii:
            allow = self.pii_detector.values(trusted_context) if trusted_context else set()
            masked = self.pii_detector.mask(output, allow=allow)
            if masked != output:
                warnings.append("PII masked in output")
                output = masked

        return output, warnings


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _redact_inputs(inputs: dict) -> dict:
    return {"text": "[REDACTED]"}


def _redact_outputs(outputs: dict) -> dict:
    return {"result": "[REDACTED]"}


class SecurityPipeline:
    """
    Full security pipeline for the Flask API.

    Put `pipeline.canary` somewhere in your system prompt (e.g. "Internal ref: <canary>")
    and check_output will block any response that echoes it.
    """

    def __init__(self, max_input_chars: int, reversible_pii: bool = False,
                 canary: Optional[str] = None, mask_pii: bool = True):
        # Pass a fixed canary (e.g. from an env var) when running several workers.
        self.canary = canary or f"CANARY-{secrets.token_hex(8)}"
        self.sanitizer = InputSanitizer(max_chars=max_input_chars)
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator(canary=self.canary, mask_pii=mask_pii)
        self.reversible_pii = reversible_pii

    # process_inputs/outputs keep raw user text and PII out of LangSmith traces.
    @traceable(name="security_check_input",
               process_inputs=_redact_inputs, process_outputs=_redact_outputs)
    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        """
        Returns: (is_allowed, cleaned_text, security_notes)
        For reversible masking use check_input_reversible().
        """
        allowed, cleaned, notes, _ = self._check_input(text)
        return allowed, cleaned, notes

    def check_input_reversible(self, text: str) -> tuple[bool, str, list[str], dict[str, str]]:
        """Like check_input but also returns the placeholder->original mapping."""
        return self._check_input(text, reversible=True)

    def _check_input(self, text: str, reversible: Optional[bool] = None):
        reversible = self.reversible_pii if reversible is None else reversible
        notes: list[str] = []

        # Normalize FIRST so the check sees what the LLM will see.
        normalized = self.sanitizer.normalize(text)

        is_safe, reason = self.sanitizer.check(normalized)
        if not is_safe:
            logger.warning("input blocked fp=%s reason=%s", _fingerprint(text), reason)
            return False, "", [reason], {}

        mapping: dict[str, str] = {}
        pii_found = self.pii_detector.detect(normalized)
        if pii_found:
            if reversible:
                normalized, mapping = self.pii_detector.mask_reversible(normalized)
            else:
                normalized = self.pii_detector.mask(normalized)
            notes.append(f"Input PII masked: {list(pii_found)}")

        return True, normalized, notes, mapping

    @traceable(name="security_check_output",
               process_inputs=_redact_inputs, process_outputs=_redact_outputs)
    def check_output(self, text: str, trusted_context: str = "") -> tuple[str, list[str]]:
        """Returns: (cleaned_output, warnings)"""
        return self.output_validator.validate(text, trusted_context)