"""Typed provenance shared by serving measurements and activation traces."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExecutionPhase = Literal["prefill", "decode", "end_to_end"]
ValueKind = Literal[
    "measurement",
    "analytical_estimate",
    "synthetic_demonstration",
    "hardware_counter_measurement",
]
TraceMode = Literal["hf_reference", "backend_native", "none"]


class MeasurementProvenance(BaseModel):
    """Identity and acquisition context for one reported observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    run_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    backend_name: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    quantization: str | None = None
    tensor_parallel_size: int = Field(gt=0)
    workload_fingerprint: str = Field(min_length=1)
    requested_prompt_tokens: int = Field(ge=0)
    actual_prompt_tokens: int | None = Field(default=None, ge=0)
    requested_output_tokens: int = Field(ge=0)
    actual_output_tokens: int | None = Field(default=None, ge=0)
    concurrency: int = Field(gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    execution_phase: ExecutionPhase
    decode_step_index: int | None = Field(default=None, ge=0)
    warmup: bool = False
    value_kind: ValueKind
    instrumentation_mode: str = Field(min_length=1)
    instrumentation_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_phase(self) -> MeasurementProvenance:
        if self.decode_step_index is not None and self.execution_phase != "decode":
            raise ValueError("decode_step_index is valid only for decode observations")
        return self


class TraceCacheIdentity(BaseModel):
    """All fields that can change the meaning of a cached activation trace."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    workload_fingerprint: str = Field(min_length=1)
    trace_mode: Literal["hf_reference", "backend_native"]
    phase: ExecutionPhase
    prompt_token_lengths: tuple[int, ...]
    output_token_lengths: tuple[int, ...]
    concurrency: int = Field(gt=0)
    top_k: int = Field(gt=0)
    tracer_version: str = Field(min_length=1)

    def digest(self) -> str:
        """Return a stable short digest suitable for a cache filename."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]
