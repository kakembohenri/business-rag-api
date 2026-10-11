from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
from .history import ConversationStore

from .config import get_settings
from .models import (
    ChatRequest,
    HealthResponse,
)
from .monitoring import RequestTimer, get_logger
from .security import SecurityPipeline
from .cache import ResponseCache
from .retriever import COMPANY_CONTACTS, NOT_FOUND_MESSAGE, runnable_chain

security: SecurityPipeline = None
cache: ResponseCache = None
agent = None
history_store= None
allowed_origins= None

MAX_INPUT_CHARS = 500

logger = get_logger()

def get_chat_rate_limit():
    return get_settings().rate_limit
    
def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global security, cache, agent, history_store, allowed_origins

    settings = get_settings()

    logger.info("Starting production API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.model_name,
        "tracing_enabled": settings.langsmith_tracing,
    }})
    history_store = ConversationStore()
    security = SecurityPipeline(canary=settings.canary, max_input_chars=MAX_INPUT_CHARS, mask_pii=False)
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    agent = runnable_chain
    allowed_origins = settings.allowed_origins

    logger.info("All components initialized. Ready to serve requests.")

    yield

    logger.info("Shutting down...")
    
    
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Business RAG LangGraph API",
    description="A production-ready chat API with security and observability.",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origins],   # never "*" in production
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    logger.warning("Rate limit exceeded", extra={"extra_data": {
        "client_ip": get_remote_address(request),
    }})
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        },
    )
    
@app.post("/chat")
@limiter.limit(get_settings().rate_limit)
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    async def event_stream():
        with RequestTimer() as timer:
            security_notes = []
            history = await history_store.get(body.thread_id)
            # Security Check
            is_allowed, cleaned_message, notes = security.check_input(body.message)
            security_notes.extend(notes)

            if not is_allowed:
                logger.warning("Request blocked by security", extra={"extra_data": {
                    "reason": notes,
                    "thread_id": body.thread_id,
                }})
                yield sse("error", {"message": "Your message was blocked by our security filters."})
                return
            yield sse("status", {"label": "Question checked"})
            
            if not history:
                cached = await cache.get(cleaned_message)
                if cached is not None:
                    logger.info("Cache hit", extra={"extra_data": {
                        "thread_id": body.thread_id,
                    }})
                    yield sse("answer", {"text": cached, "cached": True})
                    yield sse("done", {"cached": True})
                    return

            # Invoke Agent
            docs, parts = [], []
            try:
                yield sse("status", {"label": "Searching documents"})
                async for ev in agent.astream_events(
                    {"question": cleaned_message, "history":history},
                    version="v2",
                    include_types=["retriever", "chat_model"],
                    exclude_tags=["rewrite"]
                ):
                    if await request.is_disconnected():
                        return                          
                    kind = ev["event"]

                    if kind == "on_retriever_end":
                        docs = ev["data"]["output"]
                        if not docs:
                            yield sse("answer", {"text": NOT_FOUND_MESSAGE, "cached": False})
                            yield sse("done", {"latency_ms": round(timer.elapsed_ms, 2)})
                            return
                        yield sse("status", {"label": "Writing answer"})
                    elif kind == "on_chat_model_stream":
                        parts.append(ev["data"]["chunk"].content)


            except Exception as e:
                logger.error(f"Agent invocation failed: {e}", extra={"extra_data": {
                    "thread_id": body.thread_id,
                    "error": str(e),
                }})
                yield sse("error", {"message": "An error occurred while processing your request."})
                return

            # Output Validation
            validated, output_warnings = security.check_output("".join(parts), trusted_context=COMPANY_CONTACTS)
            security_notes.extend(output_warnings)

            if security_notes:
                logger.info("Security notes", extra={"extra_data": {
                    "notes": security_notes,
                    "thread_id": body.thread_id,
                }})

            if "Canary token leaked" not in output_warnings:
                await history_store.append(body.thread_id, cleaned_message, validated)
                if not history:
                    # Cache Store
                    await cache.set(cleaned_message, validated)

            logger.info("Request completed", extra={"extra_data": {
                "thread_id": body.thread_id,
                "latency_ms": round(timer.elapsed_ms, 2),
            }})

            yield sse("answer", {"text": validated, "cached": False})
            yield sse("done", {"latency_ms": round(timer.elapsed_ms, 2)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )