"""Validated application configuration models."""

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Configuration required to launch one serving backend."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    dtype: str = "auto"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    tp_size: int = Field(default=1, ge=1)
    quant: str | None = None
    startup_timeout_s: float = Field(default=600.0, gt=0)


class SweepConfig(BaseModel):
    """Shell for sweep dimensions introduced in a later build stage."""

    model_config = ConfigDict(extra="forbid")

    batch_sizes: list[int] = Field(default_factory=list)
    sequence_lengths: list[tuple[int, int]] = Field(default_factory=list)


class AppConfig(BaseModel):
    """Top-level configuration shell shared by scripts and the future CLI."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    sweep: SweepConfig | None = None
