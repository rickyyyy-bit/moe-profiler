"""Common request schema used by serving backends and workloads."""

from pydantic import BaseModel, ConfigDict, Field


class RequestSpec(BaseModel):
    """One generation request, optionally associated with a session turn."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    prompt: str
    max_tokens: int = Field(gt=0)
    arrival_time: float = Field(default=0.0, ge=0.0)
    session_id: str | None = None
    turn_index: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)
