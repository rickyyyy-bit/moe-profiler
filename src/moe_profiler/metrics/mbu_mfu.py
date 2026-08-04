"""Vanilla and sparsity-aware model bandwidth/compute utilisation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from moe_profiler.metrics.model_stats import ModelStats


def vanilla_mbu(
    stats: ModelStats,
    tpot_s: float,
    s_kv_bytes: float,
    b_peak: float,
) -> float:
    """Return vanilla model bandwidth utilisation.

    ``b_peak`` is peak memory bandwidth in bytes/second. Model weights and the
    supplied KV-cache working set are assumed to move once per output token.
    """
    _require_positive("tpot_s", tpot_s)
    _require_non_negative("s_kv_bytes", s_kv_bytes)
    _require_positive("b_peak", b_peak)
    model_bytes = stats.total_params * stats.dtype_bytes
    achieved_bandwidth = (model_bytes + s_kv_bytes) / tpot_s
    return achieved_bandwidth / b_peak


def s_mbu(
    stats: ModelStats,
    activation_matrix: ArrayLike,
    tpot_s: float,
    s_kv_bytes: float,
    b_peak: float,
) -> float:
    """Return sparsity-aware model bandwidth utilisation.

    Matrix entries are activation counts or booleans; any positive entry means
    that routed expert's weights moved. Shared experts are always active.
    """
    matrix = _validate_activation_matrix(stats, activation_matrix)
    _require_positive("tpot_s", tpot_s)
    _require_non_negative("s_kv_bytes", s_kv_bytes)
    _require_positive("b_peak", b_peak)

    if stats.n_experts == 1 and np.all(matrix > 0):
        return vanilla_mbu(stats, tpot_s, s_kv_bytes, b_peak)

    attention_params = stats.attn_flops_per_token / 2
    expert_params = _expert_params(stats)
    routed_experts_active = int(np.count_nonzero(matrix > 0))
    shared_experts_active = stats.n_layers * stats.n_shared_experts
    activated_params = (
        attention_params
        + (routed_experts_active + shared_experts_active) * expert_params
    )
    activated_bytes = activated_params * stats.dtype_bytes
    achieved_bandwidth = (activated_bytes + s_kv_bytes) / tpot_s
    return achieved_bandwidth / b_peak


def vanilla_mfu(stats: ModelStats, throughput: float, f_peak: float) -> float:
    """Return vanilla model FLOPs utilisation.

    ``throughput`` is output tokens/second and ``f_peak`` is peak FLOPs/second.
    The vanilla estimate applies the two-FLOPs-per-parameter rule to all model
    parameters.
    """
    _require_non_negative("throughput", throughput)
    _require_positive("f_peak", f_peak)
    flops_per_token = 2 * stats.total_params
    return throughput * flops_per_token / f_peak


def s_mfu(stats: ModelStats, throughput: float, f_peak: float) -> float:
    """Return sparsity-aware model FLOPs utilisation.

    Stage 3 already derives the three terms in the sparse FLOP equation from
    model configuration: attention, router, and selected/shared expert FLOPs.
    """
    _require_non_negative("throughput", throughput)
    _require_positive("f_peak", f_peak)
    sparse_flops_per_token = (
        stats.attn_flops_per_token
        + stats.router_flops_per_token
        + stats.expert_flops_per_token
    )
    return throughput * sparse_flops_per_token / f_peak


def _validate_activation_matrix(
    stats: ModelStats, activation_matrix: ArrayLike
) -> NDArray[np.generic]:
    matrix = np.asarray(activation_matrix)
    expected_shape = (stats.n_layers, stats.n_experts)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"activation_matrix shape {matrix.shape} does not match {expected_shape}"
        )
    if not (
        np.issubdtype(matrix.dtype, np.bool_) or np.issubdtype(matrix.dtype, np.number)
    ):
        raise TypeError("activation_matrix must contain booleans or numeric counts")
    if np.iscomplexobj(matrix):
        raise TypeError("activation_matrix must contain real values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("activation_matrix must contain only finite values")
    if np.any(matrix < 0):
        raise ValueError("activation_matrix cannot contain negative counts")
    return matrix


def _expert_params(stats: ModelStats) -> int:
    return 3 * stats.hidden_size * stats.expert_intermediate_size


def _require_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _require_non_negative(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
