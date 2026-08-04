"""Validated application configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """Configuration required to launch one serving backend."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    model_revision: str = Field(default="auto", min_length=1)
    dtype: str = "auto"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    tp_size: int = Field(default=1, ge=1)
    quant: str | None = None
    startup_timeout_s: float = Field(default=600.0, gt=0)


class SweepConfig(BaseModel):
    """Controlled batch-size and sequence-length sweep dimensions."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["vllm", "sglang"] = "vllm"
    batch_sizes: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32], min_length=1
    )
    sequence_lengths: list[tuple[int, int]] = Field(
        default_factory=lambda: [(128, 32), (512, 64), (2048, 128)],
        min_length=1,
    )
    device: str = "unknown"
    workload: str = "synthetic"
    results_dir: Path = Path("results")
    seed: int = 42

    @model_validator(mode="after")
    def validate_dimensions(self) -> SweepConfig:
        """Reject dimensions that cannot produce a generation request."""
        if any(batch_size <= 0 for batch_size in self.batch_sizes):
            raise ValueError("batch_sizes must contain only positive integers")
        if any(
            input_len <= 0 or output_len <= 1
            for input_len, output_len in self.sequence_lengths
        ):
            raise ValueError(
                "sequence_lengths must contain positive input lengths and output "
                "lengths greater than one"
            )
        return self


class TraceConfig(BaseModel):
    """In-process Hugging Face expert-activation tracing settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_batches: int = Field(default=8, gt=0)
    cache_dir: Path = Path("results/activations")
    device_map: str | None = "auto"
    trust_remote_code: bool = False


class AppConfig(BaseModel):
    """Top-level configuration shell shared by scripts and the future CLI."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    sweep: SweepConfig | None = None
    trace: TraceConfig | None = None
