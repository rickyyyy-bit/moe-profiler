"""Hand-calculated tests for vanilla and sparsity-aware MBU/MFU."""

import numpy as np
import pytest

from moe_profiler.metrics.mbu_mfu import s_mbu, s_mfu, vanilla_mbu, vanilla_mfu
from moe_profiler.metrics.model_stats import ModelStats


def test_vanilla_mbu_matches_model_plus_kv_byte_formula() -> None:
    stats = _stats(n_experts=8, top_k=2)
    tpot_s = 0.02
    kv_bytes = 128.0
    peak_bandwidth = 10_000.0

    result = vanilla_mbu(stats, tpot_s, kv_bytes, peak_bandwidth)

    expected = (
        (stats.total_params * stats.dtype_bytes + kv_bytes) / tpot_s
    ) / peak_bandwidth
    assert result == pytest.approx(expected)


def test_dense_all_active_reduces_to_vanilla() -> None:
    stats = _stats(n_experts=1, top_k=1)
    activation_matrix = np.ones((stats.n_layers, 1), dtype=np.int64)

    baseline_mbu = vanilla_mbu(
        stats,
        tpot_s=0.02,
        s_kv_bytes=128.0,
        b_peak=10_000.0,
    )
    sparse_mbu = s_mbu(
        stats,
        activation_matrix,
        tpot_s=0.02,
        s_kv_bytes=128.0,
        b_peak=10_000.0,
    )

    assert sparse_mbu == pytest.approx(baseline_mbu)
    assert s_mfu(stats, throughput=10.0, f_peak=100_000.0) == pytest.approx(
        vanilla_mfu(stats, throughput=10.0, f_peak=100_000.0)
    )


def test_sparse_mbu_uses_only_experts_activated_in_matrix() -> None:
    stats = _stats(n_experts=8, top_k=2)
    activation_matrix = np.zeros((stats.n_layers, stats.n_experts), dtype=np.int64)
    activation_matrix[0, :2] = [4, 3]
    activation_matrix[1, 2:4] = [2, 5]

    baseline = vanilla_mbu(stats, 0.01, 0.0, 100_000.0)
    sparse = s_mbu(stats, activation_matrix, 0.01, 0.0, 100_000.0)

    expert_params = 3 * stats.hidden_size * stats.expert_intermediate_size
    attention_params = stats.attn_flops_per_token / 2
    expected_sparse = (
        (attention_params + 4 * expert_params) * stats.dtype_bytes / 0.01
    ) / 100_000.0
    assert sparse == pytest.approx(expected_sparse)
    assert baseline > sparse
    assert baseline / sparse == pytest.approx(
        stats.n_experts / stats.top_k,
        rel=0.3,
    )


def test_sparse_mbu_counts_shared_experts_as_always_active() -> None:
    stats = _stats(n_experts=8, top_k=2, n_shared_experts=1)
    activation_matrix = np.zeros((stats.n_layers, stats.n_experts), dtype=bool)
    activation_matrix[:, :2] = True

    result = s_mbu(stats, activation_matrix, 0.01, 0.0, 100_000.0)

    expert_params = 3 * stats.hidden_size * stats.expert_intermediate_size
    attention_params = stats.attn_flops_per_token / 2
    active_experts = 4 + stats.n_layers
    expected = (
        (attention_params + active_experts * expert_params) * stats.dtype_bytes / 0.01
    ) / 100_000.0
    assert result == pytest.approx(expected)


def test_mfu_matches_hand_calculated_flop_terms() -> None:
    stats = _stats(n_experts=8, top_k=2)
    throughput = 25.0
    peak_flops = 1_000_000.0

    assert vanilla_mfu(stats, throughput, peak_flops) == pytest.approx(
        throughput * (2 * stats.total_params) / peak_flops
    )
    assert s_mfu(stats, throughput, peak_flops) == pytest.approx(
        throughput
        * (
            stats.attn_flops_per_token
            + stats.expert_flops_per_token
            + stats.router_flops_per_token
        )
        / peak_flops
    )


def test_activation_matrix_shape_and_counts_are_validated() -> None:
    stats = _stats(n_experts=8, top_k=2)

    with pytest.raises(ValueError, match="shape"):
        s_mbu(stats, np.ones((1, 8)), 0.01, 0.0, 100_000.0)
    with pytest.raises(ValueError, match="negative"):
        s_mbu(stats, -np.ones((2, 8)), 0.01, 0.0, 100_000.0)


def _stats(*, n_experts: int, top_k: int, n_shared_experts: int = 0) -> ModelStats:
    n_layers = 2
    hidden_size = 8
    expert_intermediate_size = 4
    expert_params = 3 * hidden_size * expert_intermediate_size
    attention_params = 200
    router_params = 16 if n_experts > 1 else 0
    total_params = attention_params + n_layers * n_experts * expert_params
    active_expert_params = n_layers * (top_k + n_shared_experts) * expert_params
    return ModelStats(
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_heads=2,
        n_kv_heads=1,
        head_dim=4,
        vocab_size=32,
        n_experts=n_experts,
        top_k=top_k,
        n_shared_experts=n_shared_experts,
        expert_intermediate_size=expert_intermediate_size,
        dtype_bytes=2.0,
        total_params=total_params,
        active_params_per_token=attention_params + active_expert_params + router_params,
        attn_flops_per_token=2 * attention_params,
        expert_flops_per_token=2 * active_expert_params,
        router_flops_per_token=2 * router_params,
    )
