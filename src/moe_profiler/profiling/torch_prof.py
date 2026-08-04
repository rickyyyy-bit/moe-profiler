"""Decode bandwidth measurement with explicit DRAM-byte provenance."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from moe_profiler.runner.record import RunResult

ByteSource = Literal["hardware_counter", "analytical_estimate"]

_TOTAL_BYTE_KEYS = ("dram_bytes", "dram__bytes.sum", "hbm_bytes")
_READ_BYTE_KEYS = ("dram_read_bytes", "dram__bytes_read.sum", "hbm_read_bytes")
_WRITE_BYTE_KEYS = ("dram_write_bytes", "dram__bytes_write.sum", "hbm_write_bytes")
_DEFAULT_PROFILER_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)


class ProfiledBandwidth(BaseModel):
    """Measured decode bandwidth and the source of its byte count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dram_bytes: float = Field(gt=0.0)
    elapsed_s: float = Field(gt=0.0)
    profiled_bw_gbps: float = Field(gt=0.0)
    byte_source: ByteSource


class BandwidthValidation(BaseModel):
    """Comparison between analytical S-MBU and profiled utilisation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analytic_utilization: float = Field(ge=0.0)
    profiled_utilization: float = Field(ge=0.0)
    relative_error: float = Field(ge=0.0)
    tolerance: float = Field(gt=0.0)
    within_tolerance: bool


def profile_decode_bandwidth(
    decode_step: Callable[[], object],
    *,
    bytes_per_step: float | None = None,
    warmup_steps: int = 3,
    profile_steps: int = 10,
    profiler_metrics: tuple[str, ...] | None = _DEFAULT_PROFILER_METRICS,
) -> ProfiledBandwidth:
    """Profile repeated CUDA decode steps and calculate achieved HBM bandwidth.

    PyTorch builds do not all expose CUPTI DRAM counters. When counters are not
    present, callers must provide an analytical ``bytes_per_step`` estimate;
    allocation-memory fields are intentionally not treated as DRAM traffic.
    Pass ``profiler_metrics=None`` when a CUDA/CUPTI build cannot collect the
    default per-kernel DRAM read/write metrics.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if profile_steps <= 0:
        raise ValueError("profile_steps must be positive")
    if bytes_per_step is not None:
        _positive("bytes_per_step", bytes_per_step)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for decode profiling") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decode bandwidth profiling")

    for _ in range(warmup_steps):
        decode_step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    experimental_config = None
    if profiler_metrics:
        experimental_config = torch.profiler._ExperimentalConfig(
            profiler_metrics=list(profiler_metrics),
            profiler_measure_per_kernel=True,
        )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        experimental_config=experimental_config,
    ) as profile:
        start.record()
        for _ in range(profile_steps):
            decode_step()
        end.record()
        torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0
    analytical_bytes = (
        None if bytes_per_step is None else bytes_per_step * profile_steps
    )
    return profiled_bandwidth_from_events(
        profile.events(),
        elapsed_s=elapsed_s,
        analytical_dram_bytes=analytical_bytes,
    )


def profiled_bandwidth_from_events(
    events: Iterable[object],
    *,
    elapsed_s: float,
    analytical_dram_bytes: float | None = None,
) -> ProfiledBandwidth:
    """Build a bandwidth result from profiler events or an explicit fallback."""
    _positive("elapsed_s", elapsed_s)
    counter_bytes = sum(_event_dram_bytes(event) for event in events)
    if counter_bytes > 0:
        dram_bytes = counter_bytes
        source: ByteSource = "hardware_counter"
    elif analytical_dram_bytes is not None:
        _positive("analytical_dram_bytes", analytical_dram_bytes)
        dram_bytes = analytical_dram_bytes
        source = "analytical_estimate"
    else:
        raise RuntimeError(
            "profiler events expose no DRAM-byte counters; provide bytes_per_step"
        )
    bandwidth_gbps = dram_bytes / elapsed_s / 1e9
    return ProfiledBandwidth(
        dram_bytes=dram_bytes,
        elapsed_s=elapsed_s,
        profiled_bw_gbps=bandwidth_gbps,
        byte_source=source,
    )


def validate_profiled_bandwidth(
    analytic_s_mbu: float,
    profiled_bw_gbps: float,
    peak_bw_gbps: float,
    *,
    tolerance: float = 0.05,
) -> BandwidthValidation:
    """Compare analytical utilisation with profiled bandwidth/peak bandwidth."""
    _non_negative("analytic_s_mbu", analytic_s_mbu)
    _non_negative("profiled_bw_gbps", profiled_bw_gbps)
    _positive("peak_bw_gbps", peak_bw_gbps)
    _positive("tolerance", tolerance)
    profiled_utilization = profiled_bw_gbps / peak_bw_gbps
    denominator = max(analytic_s_mbu, 1e-12)
    relative_error = abs(profiled_utilization - analytic_s_mbu) / denominator
    return BandwidthValidation(
        analytic_utilization=analytic_s_mbu,
        profiled_utilization=profiled_utilization,
        relative_error=relative_error,
        tolerance=tolerance,
        within_tolerance=relative_error <= tolerance,
    )


def record_profiled_bandwidth(
    result: RunResult, measurement: ProfiledBandwidth
) -> RunResult:
    """Return a validated RunResult carrying the profiled bandwidth value."""
    values = result.model_dump()
    values["profiled_bw_gbps"] = measurement.profiled_bw_gbps
    return RunResult.model_validate(values)


def _event_dram_bytes(event: object) -> float:
    values = _event_values(event)
    total = _first_numeric(values, _TOTAL_BYTE_KEYS)
    if total is not None:
        return total
    read = _first_numeric(values, _READ_BYTE_KEYS) or 0.0
    written = _first_numeric(values, _WRITE_BYTE_KEYS) or 0.0
    return read + written


def _event_values(event: object) -> dict[str, object]:
    values: dict[str, object] = {}
    if isinstance(event, Mapping):
        values.update(event)
    else:
        for key in (*_TOTAL_BYTE_KEYS, *_READ_BYTE_KEYS, *_WRITE_BYTE_KEYS):
            value = getattr(event, key, None)
            if value is not None:
                values[key] = value
    for container_name in ("metadata", "args", "extra_fields"):
        container = (
            event.get(container_name)
            if isinstance(event, Mapping)
            else getattr(event, container_name, None)
        )
        if isinstance(container, Mapping):
            values.update(container)
    return values


def _first_numeric(values: Mapping[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric) and numeric >= 0:
                return numeric
    return None


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _non_negative(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
