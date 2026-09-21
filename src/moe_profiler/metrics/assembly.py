"""Compatibility-gated assembly of analytical utilisation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from moe_profiler.metrics.mbu_mfu import s_mbu, s_mfu, vanilla_mbu, vanilla_mfu
from moe_profiler.metrics.model_stats import ModelStats
from moe_profiler.provenance import TraceMode


@dataclass(frozen=True)
class AssembledMetrics:
    vanilla_mbu: float | None
    s_mbu: float | None
    vanilla_mfu: float | None
    s_mfu: float | None
    status: dict[str, str]


def assemble_metrics(
    *,
    stats: ModelStats | None,
    tpot_s: float,
    throughput_tok_s: float,
    kv_bytes: float | None,
    peak_bw_bytes_s: float | None,
    peak_flops: float | None,
    activation_matrix: ArrayLike | None = None,
    trace_mode: TraceMode = "none",
    provenance_matches: bool = False,
    allow_exploratory_hf_reference: bool = False,
) -> AssembledMetrics:
    """Populate metrics only when required inputs and provenance are compatible."""
    status: dict[str, str] = {}
    if stats is None:
        reason = "unavailable: model statistics were not loaded at the serving revision"
        return AssembledMetrics(
            None,
            None,
            None,
            None,
            {
                "vanilla_mbu": reason,
                "s_mbu": reason,
                "vanilla_mfu": reason,
                "s_mfu": reason,
            },
        )

    dense_mbu = None
    sparse_mbu = None
    dense_mfu = None
    sparse_mfu = None
    if peak_bw_bytes_s is None or kv_bytes is None or tpot_s <= 0:
        status["vanilla_mbu"] = (
            "unavailable: decode TPOT, KV bytes, and peak bandwidth are required"
        )
    else:
        dense_mbu = vanilla_mbu(stats, tpot_s, kv_bytes, peak_bw_bytes_s)
        status["vanilla_mbu"] = "analytical_estimate"

    trace_allowed = trace_mode == "backend_native" and provenance_matches
    exploratory = trace_mode == "hf_reference" and allow_exploratory_hf_reference
    if activation_matrix is None:
        status["s_mbu"] = "unavailable: no decode-step activation record"
    elif not (trace_allowed or exploratory):
        status["s_mbu"] = (
            "unavailable: activation trace is not a matched backend-native decode trace"
        )
    elif peak_bw_bytes_s is None or kv_bytes is None or tpot_s <= 0:
        status["s_mbu"] = (
            "unavailable: decode TPOT, KV bytes, and peak bandwidth are required"
        )
    else:
        sparse_mbu = s_mbu(stats, activation_matrix, tpot_s, kv_bytes, peak_bw_bytes_s)
        status["s_mbu"] = (
            "exploratory_analytical_estimate_from_hf_reference"
            if exploratory
            else "analytical_estimate_from_matched_backend_trace"
        )

    if peak_flops is None:
        status["vanilla_mfu"] = "unavailable: peak FLOPs are required"
        status["s_mfu"] = "unavailable: peak FLOPs are required"
    else:
        dense_mfu = vanilla_mfu(stats, throughput_tok_s, peak_flops)
        sparse_mfu = s_mfu(stats, throughput_tok_s, peak_flops)
        status["vanilla_mfu"] = "analytical_estimate"
        status["s_mfu"] = "analytical_estimate; linear attention projections only"
    return AssembledMetrics(dense_mbu, sparse_mbu, dense_mfu, sparse_mfu, status)
