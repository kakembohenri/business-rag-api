import json
import logging
from datetime import datetime, timezone
import time

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for log aggregation (ELK, Datadog, etc.)."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        # Merge any extra data attached to the record
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)

def get_logger(name: str = "business-rag-api") -> logging.Logger:
    """Create a structured JSON logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

class RequestTimer:
    """Context manager for timing requests."""

    def __enter__(self):
        self.start = time.time()
        self.end = None
        return self

    @property
    def elapsed_ms(self) -> float:
        """Dynamically calculate elapsed time in milliseconds."""
        if self.end is not None:
            # Context manager completed; return final recorded time
            return (self.end - self.start) * 1000
        else:
            # Context manager is still running; calculate time up to this exact moment
            return (time.time() - self.start) * 1000

    def __exit__(self, *args):
        self.end = time.time()