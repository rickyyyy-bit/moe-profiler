"""Regenerate every report figure from committed, offline inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from moe_profiler.cost.devices import get_device
from moe_profiler.metrics.model_stats import ModelStats
from moe_profiler.metrics.roofline import RooflinePoint
from moe_profiler.viz.plots import (
    plot_activated_ratio_vs_batch,
    plot_kv_vs_seqlen,
    plot_throughput_vs_batch,
    plot_tpot_vs_batch,
)
from moe_profiler.viz.roofline_plot import plot_roofline

DEFAULT_CSV = Path("results/demo_sweep.csv")
DEFAULT_OUTPUT_DIR = Path("results/figures")


def regenerate_figures(
    csv_path: str | Path = DEFAULT_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Build the complete scoped figure set and return its output paths."""
    csv_input = Path(csv_path)
    if not csv_input.is_file():
        raise FileNotFoundError(f"result CSV not found: {csv_input}")
    destination = Path(output_dir)
    cache_dir = destination / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    outputs = (
        plot_throughput_vs_batch(csv_input, destination / "throughput_vs_batch.png"),
        plot_tpot_vs_batch(csv_input, destination / "tpot_vs_batch.png"),
        plot_activated_ratio_vs_batch(
            csv_input, destination / "activated_ratio_vs_batch.png"
        ),
        plot_kv_vs_seqlen(
            _demo_model_stats(),
            output_path=destination / "kv_vs_seqlen.png",
        ),
        plot_roofline(
            get_device("H100"),
            (
                RooflinePoint(
                    label="illustrative decode",
                    intensity=12.0,
                    achieved_flops=35e12,
                ),
                RooflinePoint(
                    label="illustrative prefill",
                    intensity=450.0,
                    achieved_flops=600e12,
                ),
            ),
            output_path=destination / "roofline.png",
        ),
    )
    return outputs


def _demo_model_stats() -> ModelStats:
    """Return deterministic Qwen2-MoE-like dimensions for analytical KV plots."""
    return ModelStats(
        n_layers=24,
        hidden_size=2048,
        n_heads=16,
        n_kv_heads=16,
        head_dim=128,
        vocab_size=151936,
        n_experts=60,
        top_k=4,
        n_shared_experts=1,
        expert_intermediate_size=1408,
        dtype_bytes=2.0,
        total_params=14_315_784_192,
        active_params_per_token=2_066_595_840,
        attn_flops_per_token=805_306_368,
        expert_flops_per_token=3_321_888_768,
        router_flops_per_token=5_996_544,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Regenerate all figures and print their paths."""
    arguments = _parse_args()
    for output in regenerate_figures(arguments.csv, arguments.output_dir):
        print(output)


if __name__ == "__main__":
    main()
