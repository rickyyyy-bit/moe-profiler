"""Offline tests for architecture-aware model statistics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_profiler.metrics.model_stats import (
    load_model_stats,
    model_stats_from_config,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "qwen2_moe"


def _namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def test_qwen_stats_match_hand_calculated_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    stats = load_model_stats(str(FIXTURE_DIR), dtype_bytes=2.0)

    assert stats.n_layers == 24
    assert stats.hidden_size == 2048
    assert stats.n_heads == 16
    assert stats.n_kv_heads == 16
    assert stats.head_dim == 128
    assert stats.n_experts == 60
    assert stats.top_k == 4
    assert stats.n_shared_experts == 1
    assert stats.expert_intermediate_size == 1408
    assert stats.total_params == 14_315_784_192
    assert stats.active_params_per_token == 2_066_595_840
    assert stats.attn_flops_per_token == 805_306_368
    assert stats.expert_flops_per_token == 3_321_888_768
    assert stats.router_flops_per_token == 5_996_544
    assert 0.14 < stats.active_params_per_token / stats.total_params < 0.25


@pytest.mark.parametrize(
    ("model_type", "expert_field"),
    [
        ("mixtral", {"num_local_experts": 8}),
        ("olmoe", {"num_experts": 8}),
        ("deepseek_moe", {"n_routed_experts": 8}),
    ],
)
def test_expert_count_aliases(model_type: str, expert_field: dict[str, int]) -> None:
    config = {
        "model_type": model_type,
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts_per_tok": 2,
        "tie_word_embeddings": True,
        **expert_field,
    }
    if model_type.startswith("deepseek"):
        config["first_k_dense_replace"] = 1
        config["n_shared_experts"] = 1

    stats = model_stats_from_config(_namespace(config), dtype_bytes=2.0)

    assert stats.n_experts == 8
    assert stats.top_k == 2
    assert stats.active_params_per_token < stats.total_params
    if model_type == "deepseek_moe":
        assert stats.total_params == 17_552
        assert stats.active_params_per_token == 8_448
        assert stats.attn_flops_per_token == 6_144
        assert stats.expert_flops_per_token == 9_984
        assert stats.router_flops_per_token == 768


def test_qwen_sparse_step_accounts_for_dense_mlp_layers() -> None:
    config = _namespace(
        {
            "model_type": "qwen2_moe",
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 2,
            "mlp_only_layers": [],
            "qkv_bias": True,
            "tie_word_embeddings": False,
        }
    )

    stats = model_stats_from_config(config, dtype_bytes=2.0)

    assert stats.total_params == 15_280
    assert stats.active_params_per_token == 9_376
    assert stats.attn_flops_per_token == 6_144
    assert stats.expert_flops_per_token == 12_288
    assert stats.router_flops_per_token == 320


def test_invalid_top_k_is_rejected() -> None:
    config = _namespace(
        {
            "model_type": "mixtral",
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_experts": 2,
            "num_experts_per_tok": 3,
            "tie_word_embeddings": True,
        }
    )

    with pytest.raises(ValueError, match="top_k cannot exceed n_experts"):
        model_stats_from_config(config, dtype_bytes=2.0)
