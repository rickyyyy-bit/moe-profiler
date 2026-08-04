"""Baseline throughput, latency, and KV-cache plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from moe_profiler.metrics.kv_cache import kv_cache_bytes
from moe_profiler.metrics.model_stats import ModelStats


def plot_throughput_vs_batch(
    csv_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Plot output-token throughput against batch size for each sequence pair."""
    frame = _load_results(csv_path, required={"throughput_tok_s"})
    output = _result_plot_path(csv_path, output_path, "throughput_vs_batch")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for (input_len, output_len), group in frame.groupby(
        ["input_len", "output_len"], sort=True
    ):
        ordered = group.sort_values("batch_size")
        axis.plot(
            ordered["batch_size"],
            ordered["throughput_tok_s"],
            marker="o",
            label=f"in={input_len}, out={output_len}",
        )
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Throughput (output tokens/s)")
    axis.set_title("Serving throughput vs batch size")
    axis.grid(alpha=0.3)
    axis.legend()
    return _save_figure(figure, output, plt)


def plot_tpot_vs_batch(
    csv_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Plot mean time per output token against batch size."""
    frame = _load_results(csv_path, required={"tpot_s"})
    output = _result_plot_path(csv_path, output_path, "tpot_vs_batch")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for (input_len, output_len), group in frame.groupby(
        ["input_len", "output_len"], sort=True
    ):
        ordered = group.sort_values("batch_size")
        axis.plot(
            ordered["batch_size"],
            ordered["tpot_s"],
            marker="o",
            label=f"in={input_len}, out={output_len}",
        )
    axis.set_xlabel("Batch size")
    axis.set_ylabel("TPOT (seconds/token)")
    axis.set_title("Time per output token vs batch size")
    axis.grid(alpha=0.3)
    axis.legend()
    return _save_figure(figure, output, plt)


def plot_kv_vs_seqlen(
    stats: ModelStats,
    seq_lengths: Sequence[int] | None = None,
    *,
    batch_size: int = 1,
    output_path: str | Path = "results/figures/kv_vs_seqlen.png",
) -> Path:
    """Plot analytical KV-cache growth for a model and fixed batch size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    lengths = list(seq_lengths or [128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("seq_lengths must contain only positive integers")
    kv_gib = [
        kv_cache_bytes(stats, seq_len=length, batch=batch_size) / (1024**3)
        for length in lengths
    ]
    output = Path(output_path)
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(lengths, kv_gib, marker="o")
    axis.set_xlabel("Sequence length (tokens)")
    axis.set_ylabel("KV cache (GiB)")
    axis.set_title(f"Analytical KV-cache growth (batch={batch_size})")
    axis.grid(alpha=0.3)
    return _save_figure(figure, output, plt)


def _load_results(csv_path: str | Path, required: set[str]) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to plot sweep results") from exc
    path = Path(csv_path)
    frame = pd.read_csv(path)
    base_columns = {"batch_size", "input_len", "output_len"}
    missing = (base_columns | required) - set(frame.columns)
    if missing:
        raise ValueError(f"results CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("results CSV contains no rows")
    return frame


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate plots") from exc
    return plt


def _result_plot_path(
    csv_path: str | Path,
    output_path: str | Path | None,
    suffix: str,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    csv_stem = Path(csv_path).stem
    return Path("results/figures") / f"{csv_stem}_{suffix}.png"


def _save_figure(figure: Any, output: Path, plt: Any) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output
