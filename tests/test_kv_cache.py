"""Tests for the exact analytical KV-cache byte formula."""

import pytest

from moe_profiler.metrics.kv_cache import kv_cache_bytes, kv_cache_bytes_per_token
from moe_profiler.metrics.model_stats import ModelStats


@pytest.fixture
def stats() -> ModelStats:
    return ModelStats(
        n_layers=4,
        hidden_size=16,
        n_heads=4,
        n_kv_heads=2,
        head_dim=4,
        vocab_size=128,
        n_experts=4,
        top_k=2,
        n_shared_experts=0,
        expert_intermediate_size=8,
        dtype_bytes=2.0,
        total_params=20_000,
        active_params_per_token=8_000,
        attn_flops_per_token=1_000,
        expert_flops_per_token=2_000,
        router_flops_per_token=100,
    )


def test_kv_cache_bytes_match_hand_calculation(stats: ModelStats) -> None:
    # 2 (K/V) * 4 layers * 2 KV heads * 4 head dim * 2 bytes = 128.
    assert kv_cache_bytes_per_token(stats) == 128
    assert kv_cache_bytes(stats, seq_len=16, batch=3) == 6_144


def test_kv_cache_supports_sub_byte_dtypes(stats: ModelStats) -> None:
    int4_stats = stats.model_copy(update={"dtype_bytes": 0.5})

    assert kv_cache_bytes_per_token(int4_stats) == 32


def test_kv_cache_rejects_invalid_dimensions(stats: ModelStats) -> None:
    with pytest.raises(ValueError, match="seq_len"):
        kv_cache_bytes(stats, seq_len=-1, batch=1)
    with pytest.raises(ValueError, match="batch"):
        kv_cache_bytes(stats, seq_len=1, batch=0)
