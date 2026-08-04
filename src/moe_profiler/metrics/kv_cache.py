"""Analytical key-value cache memory calculations."""

from moe_profiler.metrics.model_stats import ModelStats


def kv_cache_bytes_per_token(stats: ModelStats) -> float:
    """Return KV-cache bytes for one token across every transformer layer."""
    return 2 * stats.n_layers * stats.n_kv_heads * stats.head_dim * stats.dtype_bytes


def kv_cache_bytes(stats: ModelStats, seq_len: int, batch: int) -> float:
    """Return KV-cache bytes for ``batch`` sequences of ``seq_len`` tokens."""
    if seq_len < 0:
        raise ValueError("seq_len cannot be negative")
    if batch <= 0:
        raise ValueError("batch must be positive")
    return kv_cache_bytes_per_token(stats) * seq_len * batch
