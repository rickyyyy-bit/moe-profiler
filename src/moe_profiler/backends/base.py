"""Backend contract and shared generation result schema."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from moe_profiler.workloads.base import RequestSpec

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class ExpertTraceRecorder(Protocol):
    """Contract for in-process expert tracing, separate from serving backends."""

    def attach(self, model: object) -> None:
        """Attach router hooks to an in-process model."""

    def run(
        self,
        model: object,
        tokenizer: object,
        dataset: Iterable[object],
        max_batches: int,
        *,
        batch_size: int = 1,
        model_id: str | None = None,
    ) -> "NDArray[np.int64]":
        """Trace representative batches and return per-expert counts."""

    def activation_matrix(self) -> "NDArray[np.int64]":
        """Return an ``[n_layers, n_experts]`` activation-count matrix."""

    def activated_ratio(self, batch_size: int) -> float:
        """Return the fraction of routed and shared experts activated."""


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
