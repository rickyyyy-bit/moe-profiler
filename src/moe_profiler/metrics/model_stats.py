"""Derive MoE shapes, parameter counts, and per-token FLOPs from HF configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelStats(BaseModel):
    """Architecture statistics needed by later sparsity-aware metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    n_experts: int = Field(gt=0)
    top_k: int = Field(gt=0)
    n_shared_experts: int = Field(ge=0)
    expert_intermediate_size: int = Field(gt=0)
    dtype_bytes: float = Field(gt=0.0)
    total_params: int = Field(gt=0)
    active_params_per_token: int = Field(gt=0)
    attn_flops_per_token: int = Field(ge=0)
    expert_flops_per_token: int = Field(ge=0)
    router_flops_per_token: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_routing(self) -> ModelStats:
        """Ensure the routing shape is physically meaningful."""
        if self.top_k > self.n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        if self.active_params_per_token > self.total_params:
            raise ValueError("active parameters cannot exceed total parameters")
        return self


class _ConfigLike(Protocol):
    """Structural type for a Hugging Face configuration object."""

    def __getattr__(self, name: str) -> object: ...


@dataclass(frozen=True)
class _LayerCounts:
    attention_linear: int
    attention_non_linear: int
    expert_linear: int
    router_linear: int
    other: int

    @property
    def total(self) -> int:
        return (
            self.attention_linear
            + self.attention_non_linear
            + self.expert_linear
            + self.router_linear
            + self.other
        )


def load_model_stats(model_id: str, dtype_bytes: float) -> ModelStats:
    """Load a Hugging Face config and derive analytical model statistics.

    Only configuration metadata is downloaded. Model weights are never loaded.
    ``HF_TOKEN`` is picked up by Transformers through its normal environment
    handling when access to a gated model is required.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required to load model statistics; install the "
            "dependencies from requirements.txt."
        ) from exc

    config = AutoConfig.from_pretrained(model_id)
    return model_stats_from_config(config, dtype_bytes=dtype_bytes)


def model_stats_from_config(config: _ConfigLike, dtype_bytes: float) -> ModelStats:
    """Derive statistics from an already loaded Transformers config.

    This public helper keeps offline fixtures deterministic and is also useful
    when callers already hold an ``AutoConfig`` instance.
    """
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")

    n_layers = _required_int(config, "num_hidden_layers", "n_layer")
    hidden_size = _required_int(config, "hidden_size", "d_model")
    n_heads = _required_int(config, "num_attention_heads", "n_head")
    n_kv_heads = _optional_int(
        config, "num_key_value_heads", "num_kv_heads", default=n_heads
    )
    head_dim = _optional_int(config, "head_dim", default=hidden_size // n_heads)
    vocab_size = _required_int(config, "vocab_size")
    n_experts = _optional_int(
        config,
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
        "n_experts",
        default=1,
    )
    top_k = _optional_int(
        config,
        "num_experts_per_tok",
        "num_selected_experts",
        "moe_top_k",
        "top_k",
        default=1,
    )
    expert_intermediate_size = _optional_int(
        config,
        "moe_intermediate_size",
        "expert_intermediate_size",
        "intermediate_size",
        "ffn_dim",
    )
    dense_intermediate_size = _optional_int(
        config,
        "intermediate_size",
        "ffn_dim",
        default=expert_intermediate_size,
    )
    n_shared_experts = _shared_expert_count(config)
    model_type = str(_value(config, "model_type", default="")).lower()

    sparse_layers = _sparse_layer_indices(config, n_layers, n_experts, model_type)
    attention_linear, attention_non_linear = _attention_params(
        config,
        model_type=model_type,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    expert_params = 3 * hidden_size * expert_intermediate_size
    dense_mlp_params = 3 * hidden_size * dense_intermediate_size
    shared_expert_params, shared_gate_params = _shared_expert_params(
        config,
        model_type=model_type,
        hidden_size=hidden_size,
        expert_intermediate_size=expert_intermediate_size,
        n_shared_experts=n_shared_experts,
    )
    layer_norm_params = 2 * hidden_size

    total_layer_params = 0
    active_attention_params = 0
    active_expert_params = 0
    active_router_params = 0

    for layer_index in range(n_layers):
        if layer_index in sparse_layers:
            router_params = hidden_size * n_experts + shared_gate_params
            layer_counts = _LayerCounts(
                attention_linear=attention_linear,
                attention_non_linear=attention_non_linear,
                expert_linear=n_experts * expert_params + shared_expert_params,
                router_linear=router_params,
                other=layer_norm_params,
            )
            active_expert_params += top_k * expert_params + shared_expert_params
            active_router_params += router_params
        else:
            layer_counts = _LayerCounts(
                attention_linear=attention_linear,
                attention_non_linear=attention_non_linear,
                expert_linear=dense_mlp_params,
                router_linear=0,
                other=layer_norm_params,
            )
            active_expert_params += dense_mlp_params

        total_layer_params += layer_counts.total
        active_attention_params += attention_linear

    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if _bool(config, "tie_word_embeddings") else embedding_params
    final_norm_params = hidden_size
    total_params = (
        embedding_params + total_layer_params + final_norm_params + lm_head_params
    )
    active_params = (
        active_attention_params + active_expert_params + active_router_params
    )

    return ModelStats(
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        n_experts=n_experts,
        top_k=top_k,
        n_shared_experts=n_shared_experts,
        expert_intermediate_size=expert_intermediate_size,
        dtype_bytes=dtype_bytes,
        total_params=total_params,
        active_params_per_token=active_params,
        attn_flops_per_token=2 * active_attention_params,
        expert_flops_per_token=2 * active_expert_params,
        router_flops_per_token=2 * active_router_params,
    )


def _attention_params(
    config: _ConfigLike,
    *,
    model_type: str,
    hidden_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    """Return linear and auxiliary parameters for one attention block."""
    kv_lora_rank = _maybe_int(config, "kv_lora_rank")
    qk_rope_head_dim = _maybe_int(config, "qk_rope_head_dim")
    qk_nope_head_dim = _maybe_int(config, "qk_nope_head_dim")
    value_head_dim = _maybe_int(config, "v_head_dim")
    if (
        kv_lora_rank is not None
        and qk_rope_head_dim is not None
        and qk_nope_head_dim is not None
        and value_head_dim is not None
    ):
        return _deepseek_attention_params(
            config,
            hidden_size=hidden_size,
            n_heads=n_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            value_head_dim=value_head_dim,
        )

    query_size = n_heads * head_dim
    kv_size = n_kv_heads * head_dim
    linear = (
        hidden_size * query_size + 2 * hidden_size * kv_size + query_size * hidden_size
    )
    auxiliary = 0
    if _bool(config, "qkv_bias"):
        auxiliary += query_size + 2 * kv_size
    elif _bool(config, "attention_bias"):
        auxiliary += query_size + 2 * kv_size + hidden_size
    if model_type == "olmoe":
        auxiliary += hidden_size + kv_size
    return linear, auxiliary


def _deepseek_attention_params(
    config: _ConfigLike,
    *,
    hidden_size: int,
    n_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    value_head_dim: int,
) -> tuple[int, int]:
    qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
    q_lora_rank = _maybe_int(config, "q_lora_rank")
    attention_bias = _bool(config, "attention_bias")
    if q_lora_rank is None:
        query_params = hidden_size * n_heads * qk_head_dim
        query_norm_params = 0
        query_bias_params = 0
    else:
        query_params = hidden_size * q_lora_rank + q_lora_rank * n_heads * qk_head_dim
        query_norm_params = q_lora_rank
        query_bias_params = q_lora_rank if attention_bias else 0

    kv_a_output = kv_lora_rank + qk_rope_head_dim
    linear = (
        query_params
        + hidden_size * kv_a_output
        + kv_lora_rank * n_heads * (qk_nope_head_dim + value_head_dim)
        + n_heads * value_head_dim * hidden_size
    )
    auxiliary = query_norm_params + kv_lora_rank + query_bias_params
    if attention_bias:
        auxiliary += kv_a_output + hidden_size
    return linear, auxiliary


def _shared_expert_count(config: _ConfigLike) -> int:
    explicit = _maybe_int(config, "n_shared_experts", "num_shared_experts")
    if explicit is not None:
        return explicit
    shared_size = _maybe_int(config, "shared_expert_intermediate_size")
    return 1 if shared_size is not None and shared_size > 0 else 0


def _shared_expert_params(
    config: _ConfigLike,
    *,
    model_type: str,
    hidden_size: int,
    expert_intermediate_size: int,
    n_shared_experts: int,
) -> tuple[int, int]:
    if n_shared_experts == 0:
        return 0, 0
    shared_intermediate_size = _maybe_int(config, "shared_expert_intermediate_size")
    if shared_intermediate_size is None:
        shared_intermediate_size = expert_intermediate_size * n_shared_experts
    shared_params = 3 * hidden_size * shared_intermediate_size
    shared_gate_params = hidden_size if model_type == "qwen2_moe" else 0
    return shared_params, shared_gate_params


def _sparse_layer_indices(
    config: _ConfigLike,
    n_layers: int,
    n_experts: int,
    model_type: str,
) -> set[int]:
    if n_experts <= 1:
        return set()
    if model_type == "qwen2_moe":
        dense_layers = set(_int_list(config, "mlp_only_layers"))
        sparse_step = _optional_int(config, "decoder_sparse_step", default=1)
        return {
            index
            for index in range(n_layers)
            if index not in dense_layers and (index + 1) % sparse_step == 0
        }
    if model_type.startswith("deepseek"):
        first_sparse_layer = _optional_int(config, "first_k_dense_replace", default=0)
        sparse_frequency = _optional_int(config, "moe_layer_freq", default=1)
        return {
            index
            for index in range(n_layers)
            if index >= first_sparse_layer
            and (index - first_sparse_layer) % sparse_frequency == 0
        }
    return set(range(n_layers))


def _required_int(config: _ConfigLike, *names: str) -> int:
    value = _maybe_int(config, *names)
    if value is None or value <= 0:
        joined_names = "/".join(names)
        raise ValueError(f"config requires a positive {joined_names}")
    return value


def _optional_int(config: _ConfigLike, *names: str, default: int | None = None) -> int:
    value = _maybe_int(config, *names)
    if value is None:
        if default is None:
            joined_names = "/".join(names)
            raise ValueError(f"config requires {joined_names}")
        value = default
    if value <= 0:
        joined_names = "/".join(names)
        raise ValueError(f"config requires a positive {joined_names}")
    return value


def _maybe_int(config: _ConfigLike, *names: str) -> int | None:
    value = _value(config, *names, default=None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        joined_names = "/".join(names)
        raise ValueError(f"config field {joined_names} must be an integer")
    return value


def _int_list(config: _ConfigLike, name: str) -> list[int]:
    value = _value(config, name, default=[])
    if value is None:
        return []
    if not isinstance(value, list | tuple) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"config field {name} must be a list of integers")
    return list(value)


def _bool(config: _ConfigLike, name: str) -> bool:
    value = _value(config, name, default=False)
    if not isinstance(value, bool):
        raise ValueError(f"config field {name} must be a boolean")
    return value


def _value(config: _ConfigLike, *names: str, default: object) -> object:
    for name in names:
        try:
            value = getattr(config, name)
        except AttributeError:
            continue
        if value is not None:
            return value
    return default
