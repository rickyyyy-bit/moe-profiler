"""Offline integration tests for Stage 4 sweeps, CSV logging, and plots."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from moe_profiler.backends.base import Backend, GenerationResult
from moe_profiler.config import AppConfig, ModelConfig, SweepConfig
from moe_profiler.metrics.model_stats import ModelStats
from moe_profiler.runner.record import RunResult
from moe_profiler.runner.sweep import load_sweep_config, run_sweep
from moe_profiler.viz.plots import (
    plot_kv_vs_seqlen,
    plot_throughput_vs_batch,
    plot_tpot_vs_batch,
)
from moe_profiler.workloads.base import RequestSpec


def test_default_sweep_config_loads() -> None:
    config = load_sweep_config("configs/sweep_default.yaml")

    assert config.sweep is not None
    assert config.sweep.batch_sizes == [1, 2, 4, 8, 16, 32]
    assert config.sweep.sequence_lengths[0] == (128, 32)


def test_sweep_writes_full_schema_and_generates_plots() -> None:
    with TemporaryDirectory(prefix="stage4-test-", dir="results") as directory:
        temp_path = Path(directory)
        config = AppConfig(
            model=ModelConfig(model_id="offline/test-model", dtype="bfloat16"),
            sweep=SweepConfig(
                backend="vllm",
                batch_sizes=[1, 2, 4],
                sequence_lengths=[(8, 4), (16, 4)],
                results_dir=temp_path / "results",
            ),
        )
        backend = _FakeBackend()
        csv_path = run_sweep(
            config,
            backend=backend,
            clock=_StepClock(),
            revision_resolver=lambda model_id, revision: "a" * 40,
        )

        assert backend.started is True
        assert backend.stopped is True
        frame = pd.read_csv(csv_path)
        assert list(frame.columns) == list(RunResult.model_fields)
        assert len(frame) == 6
        assert set(frame["batch_size"]) == {1, 2, 4}
        assert frame["throughput_tok_s"].gt(0).all()
        assert set(frame["model_revision"]) == {"a" * 40}
        for _, group in frame.groupby(["input_len", "output_len"]):
            assert group.sort_values("batch_size")[
                "throughput_tok_s"
            ].is_monotonic_increasing

        figure_dir = temp_path / "figures"
        throughput_plot = plot_throughput_vs_batch(
            csv_path, figure_dir / "throughput.png"
        )
        tpot_plot = plot_tpot_vs_batch(csv_path, figure_dir / "tpot.png")
        kv_plot = plot_kv_vs_seqlen(
            _stats(),
            [8, 16, 32],
            output_path=figure_dir / "kv.png",
        )
        for plot in (throughput_plot, tpot_plot, kv_plot):
            assert plot.exists()
            assert plot.stat().st_size > 0


class _FakeBackend(Backend):
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(
        self,
        model_id: str,
        quant: str | None = None,
        tp_size: int = 1,
        **kwargs: object,
    ) -> None:
        del model_id, quant, tp_size, kwargs
        self.started = True

    def generate(self, requests: list[RequestSpec]) -> list[GenerationResult]:
        return [
            GenerationResult(
                request_id=request.request_id,
                prompt_tokens=int(request.metadata["target_input_len"]),
                output_tokens=request.max_tokens,
                ttft_s=0.01,
                tpot_s=0.002,
                e2e_s=0.02,
                output_text="test",
            )
            for request in requests
        ]

    def stop(self) -> None:
        self.stopped = True

    @property
    def version(self) -> str:
        return "test-version"


class _StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


def _stats() -> ModelStats:
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
