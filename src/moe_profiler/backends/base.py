"""Backend contract and shared generation result schema."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from moe_profiler.workloads.base import RequestSpec


class BackendError(RuntimeError):
    """Base exception for serving backend failures."""


class BackendUnavailableError(BackendError):
    """Raised when a requested serving backend cannot be used."""


class GenerationResult(BaseModel):
    """Latency and token counts measured for one completed request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    ttft_s: float = Field(ge=0.0)
    tpot_s: float = Field(ge=0.0)
    e2e_s: float = Field(ge=0.0)
    output_text: str


class Backend(ABC):
    """Interface implemented by OpenAI-compatible model servers."""

    @abstractmethod
    def start(
        self,
        model_id: str,
        quant: str | None = None,
        tp_size: int = 1,
        **kwargs: object,
    ) -> None:
        """Start the backend and wait until it is ready."""

    @abstractmethod
    def generate(self, requests: list[RequestSpec]) -> list[GenerationResult]:
        """Generate responses and return one result per request."""

    @abstractmethod
    def stop(self) -> None:
        """Stop resources owned by the backend."""

    def supports_expert_recording(self) -> bool:
        """Return whether this serving backend exposes expert traces."""
        return False

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the installed backend version without importing it eagerly."""
