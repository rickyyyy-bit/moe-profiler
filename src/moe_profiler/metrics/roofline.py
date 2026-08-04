"""Roofline arithmetic and memory/compute-bound classification."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from moe_profiler.cost.devices import DeviceSpec, Precision

Bound = Literal["memory", "compute"]


class RooflinePoint(BaseModel):
    """One measured workload point for a roofline plot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    intensity: float = Field(gt=0.0)
    achieved_flops: float = Field(gt=0.0)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """Return FLOPs per byte moved."""
    _positive("flops", flops)
    _positive("bytes_moved", bytes_moved)
    return flops / bytes_moved


def ridge_point(device: DeviceSpec, precision: Precision = "fp16") -> float:
    """Return the arithmetic intensity where memory and compute roofs meet."""
    peak_bandwidth_bytes_s = device.peak_bw_gbps * 1e9
    return device.peak_flops(precision) / peak_bandwidth_bytes_s


def classify(
    point: RooflinePoint | float,
    device: DeviceSpec,
    precision: Precision = "fp16",
) -> Bound:
    """Classify a point/intensity relative to the device ridge point."""
    intensity = point.intensity if isinstance(point, RooflinePoint) else point
    _positive("intensity", intensity)
    return "memory" if intensity < ridge_point(device, precision) else "compute"


def roof_performance(
    intensity: float,
    device: DeviceSpec,
    precision: Precision = "fp16",
) -> float:
    """Return the attainable roof in operations/second at ``intensity``."""
    _positive("intensity", intensity)
    memory_roof = intensity * device.peak_bw_gbps * 1e9
    return min(memory_roof, device.peak_flops(precision))


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
