"""Offline provenance, workload, and metric-compatibility tests."""

from __future__ import annotations

import numpy as np
import pytest

from moe_profiler.metrics.assembly import assemble_metrics
from moe_profiler.metrics.model_stats import ModelStats
from moe_profiler.provenance import TraceCacheIdentity
from moe_profiler.workloads.base import build_exact_length_manifest


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return " ".join(f"token-{token}" for token in token_ids)


def test_manifest_has_exact_lengths_and_stable_fingerprint() -> None:
    kwargs = {
        "tokenizer": _Tokenizer(),
        "tokenizer_id": "offline/tokenizer",
        "tokenizer_revision": "a" * 40,
        "request_prefix": "stable",
        "concurrency": 3,
        "prompt_tokens": 11,
        "output_tokens": 5,
    }
    first = build_exact_length_manifest(**kwargs)
    second = build_exact_length_manifest(**kwargs)

    assert first.fingerprint == second.fingerprint
    assert [request.actual_prompt_tokens for request in first.requests] == [11] * 3
    assert all(len(request.prompt_token_ids) == 11 for request in first.requests)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "d" * 40),
        ("tokenizer_revision", "e" * 40),
        ("dtype", "float16"),
        ("workload_fingerprint", "f" * 64),
        ("trace_mode", "backend_native"),
        ("phase", "decode"),
        ("prompt_token_lengths", (7, 7)),
        ("output_token_lengths", (9, 9)),
        ("concurrency", 2),
        ("top_k", 1),
        ("tracer_version", "3"),
    ],
)
def test_every_important_cache_field_changes_identity(
    field: str, value: object
) -> None:
    identity = _identity()
    changed = identity.model_copy(update={field: value})

    assert identity.digest() != changed.digest()


def test_hf_reference_cannot_silently_produce_s_mbu() -> None:
    stats = _stats()
    matrix = np.ones((2, 4), dtype=np.int64)

    result = assemble_metrics(
        stats=stats,
        tpot_s=0.01,
        throughput_tok_s=100,
        kv_bytes=128,
        peak_bw_bytes_s=1_000_000,
        peak_flops=1_000_000,
        activation_matrix=matrix,
        trace_mode="hf_reference",
        provenance_matches=True,
    )

    assert result.s_mbu is None
    assert "not a matched backend-native" in result.status["s_mbu"]


def _identity() -> TraceCacheIdentity:
    return TraceCacheIdentity(
        model_id="offline/model",
        model_revision="a" * 40,
        tokenizer_id="offline/tokenizer",
        tokenizer_revision="b" * 40,
        dtype="bfloat16",
        workload_fingerprint="c" * 64,
        trace_mode="hf_reference",
        phase="prefill",
        prompt_token_lengths=(8,),
        output_token_lengths=(4,),
        concurrency=1,
        top_k=2,
        tracer_version="2",
    )


def _stats() -> ModelStats:
    return ModelStats(
        n_layers=2,
        hidden_size=8,
        n_heads=2,
        n_kv_heads=1,
        head_dim=4,
        vocab_size=32,
        n_experts=4,
        top_k=2,
        n_shared_experts=0,
        expert_intermediate_size=4,
        dtype_bytes=2,
        total_params=10_000,
        active_params_per_token=2_000,
        attn_flops_per_token=400,
        expert_flops_per_token=800,
        router_flops_per_token=32,
    )
