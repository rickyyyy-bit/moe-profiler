"""Command-line entry point for reproducible profiler workflows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from moe_profiler.metrics.model_stats import model_stats_from_config
from moe_profiler.runner.sweep import (
    load_sweep_config,
    measure_activation_ratios,
    resolve_model_revision,
    run_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="moe-profiler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("offline-check", help="run model-accounting checks")
    for command in ("sweep", "hf-trace"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
    report = subparsers.add_parser("report")
    report.add_argument("csv", type=Path)
    subparsers.add_parser(
        "backend-trace", help="explain backend-native trace availability"
    )
    args = parser.parse_args()

    if args.command == "offline-check":
        _offline_check()
    elif args.command == "sweep":
        print(run_sweep(args.config))
    elif args.command == "hf-trace":
        config = load_sweep_config(args.config)
        revision = resolve_model_revision(
            config.model.model_id, config.model.model_revision
        )
        print(measure_activation_ratios(config, revision))
    elif args.command == "report":
        _validate_report(args.csv)
    else:
        raise SystemExit(
            "Backend-native expert tracing is unsupported in the current vLLM and "
            "SGLang adapters. No trace will be simulated."
        )


def _offline_check() -> None:
    config = SimpleNamespace(
        model_type="mixtral",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=True,
    )
    stats = model_stats_from_config(config, dtype_bytes=2.0)
    print(
        "offline accounting OK: "
        f"total_params={stats.total_params}, sparse_layers={stats.sparse_layer_indices}"
    )


def _validate_report(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise SystemExit(f"report input {path} contains no rows")
    invalid = [
        row
        for row in rows
        if row.get("s_mbu") not in {None, "", "nan"}
        and row.get("trace_mode") != "backend_native"
        and "exploratory" not in row.get("metric_status_json", "")
    ]
    if invalid:
        raise SystemExit(
            "report refused: S-MBU rows use an unmatched or unspecified "
            "activation trace"
        )
    print(f"report input valid: {len(rows)} rows from {path}")


if __name__ == "__main__":
    main()
