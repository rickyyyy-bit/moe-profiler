"""Roofline plotting for measured prefill and decode workload points."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from moe_profiler.cost.devices import DeviceSpec, Precision
from moe_profiler.metrics.roofline import RooflinePoint, classify, ridge_point


def plot_roofline(
    device: DeviceSpec,
    points: Sequence[RooflinePoint],
    *,
    precision: Precision = "fp16",
    output_path: str | Path = "results/figures/roofline.png",
) -> Path:
    """Plot memory/compute roofs and labelled workload points."""
    if not points:
        raise ValueError("points cannot be empty")
    ridge = ridge_point(device, precision)
    point_intensities = [point.intensity for point in points]
    lower = min(0.1, min(point_intensities) / 5)
    upper = max(ridge * 20, max(point_intensities) * 5)
    intensities = np.logspace(np.log10(lower), np.log10(upper), 300)
    peak_bandwidth = device.peak_bw_gbps * 1e9
    peak_flops = device.peak_flops(precision)
    memory_roof = intensities * peak_bandwidth
    compute_roof = np.full_like(intensities, peak_flops)

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    axis.loglog(
        intensities,
        memory_roof / 1e12,
        linestyle="--",
        label=f"Memory roof ({device.peak_bw_gbps:g} GB/s)",
    )
    axis.loglog(
        intensities,
        compute_roof / 1e12,
        linestyle="--",
        label=f"Compute roof ({peak_flops / 1e12:g} TFLOPs/s)",
    )
    axis.loglog(
        intensities,
        np.minimum(memory_roof, compute_roof) / 1e12,
        linewidth=2,
        color="black",
        label="Attainable roof",
    )
    for point in points:
        bound = classify(point, device, precision)
        axis.scatter(point.intensity, point.achieved_flops / 1e12, s=60, zorder=3)
        axis.annotate(
            f"{point.label} ({bound}-bound)",
            (point.intensity, point.achieved_flops / 1e12),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.axvline(ridge, color="grey", alpha=0.5, linewidth=1)
    axis.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    axis.set_ylabel("Performance (TFLOPs/s)")
    axis.set_title(f"{device.name} {precision.upper()} roofline")
    axis.grid(which="both", alpha=0.25)
    axis.legend()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate roofline plots") from exc
    return plt
