"""Serving backend interfaces."""

from typing import Any

from moe_profiler.backends.base import (
    Backend,
    BackendError,
    BackendUnavailableError,
    GenerationResult,
)


def get_backend(name: str, **options: Any) -> Backend:
    """Create a serving backend by name without eagerly importing engines."""
    normalized_name = name.strip().lower()
    if normalized_name == "vllm":
        from moe_profiler.backends.vllm_backend import VllmBackend

        return VllmBackend(**options)
    if normalized_name == "sglang":
        from moe_profiler.backends.sglang_backend import SglangBackend

        return SglangBackend(**options)
    raise ValueError(f"unsupported backend {name!r}; expected one of: 'vllm', 'sglang'")


__all__ = [
    "Backend",
    "BackendError",
    "BackendUnavailableError",
    "GenerationResult",
    "get_backend",
]
