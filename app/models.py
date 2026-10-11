"""
API Request and Response Models
Pydantic models for input validation and response structure.
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The user's message to the agent",
    )
    thread_id: str = Field(default="default", description="Conversation thread ID")

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    environment: str
    version: str = "1.1.0"
    checks: dict = {}

class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None