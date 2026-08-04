"""Offline tests for Stage 6 activation-ratio sweep integration."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from moe_profiler.backends.base import Backend, GenerationResult
from moe_profiler.config import AppConfig, ModelConfig, SweepConfig, TraceConfig
from moe_profiler.runner.sweep import run_sweep
from moe_profiler.viz.plots import plot_activated_ratio_vs_batch
from moe_profiler.workloads.base import RequestSpec


def test_sweep_records_batch_ratios_and_generates_sparsity_plot() -> None:
    with TemporaryDirectory(prefix="stage6-test-", dir="results") as directory:
        backend = _FakeBackend()
        provider_called = False

        def ratio_provider(config: AppConfig, revision: str) -> dict[int, float]:
            nonlocal provider_called
            assert backend.started is False
            assert config.trace is not None
            assert revision == "b" * 40
            provider_called = True
            return {1: 0.2, 4: 0.55, 32: 0.8}

        root = Path(directory)
        config = AppConfig(
            model=ModelConfig(model_id="offline/fake-moe", dtype="bfloat16"),
            sweep=SweepConfig(
                batch_sizes=[1, 4, 32],
                sequence_lengths=[(8, 4), (16, 4)],
                results_dir=root / "results",
            ),
            trace=TraceConfig(max_batches=3, cache_dir=root / "activations"),
        )

        csv_path = run_sweep(
            config,
            backend=backend,
            clock=_StepClock(),
            revision_resolver=lambda model_id, revision: "b" * 40,
            activation_ratio_provider=ratio_provider,
        )

        assert provider_called is True
        assert backend.started is True
        assert backend.stopped is True
        frame = pd.read_csv(csv_path)
        expected = {1: 0.2, 4: 0.55, 32: 0.8}
        for batch_size, ratio in expected.items():
            rows = frame.loc[frame["batch_size"] == batch_size]
            assert rows["activated_param_ratio"].eq(ratio).all()

        plot = plot_activated_ratio_vs_batch(
            csv_path, root / "figures" / "activated_ratio.png"
        )
        assert plot.exists()
        assert plot.stat().st_size > 0


def test_trace_provider_must_cover_every_batch_size() -> None:
    config = AppConfig(
        model=ModelConfig(model_id="offline/fake-moe"),
        sweep=SweepConfig(batch_sizes=[1, 2], sequence_lengths=[(8, 4)]),
        trace=TraceConfig(),
    )
    backend = _FakeBackend()

    with pytest.raises(ValueError, match=r"omitted batch sizes: \[2\]"):
        run_sweep(
            config,
            backend=backend,
            revision_resolver=lambda model_id, revision: "c" * 40,
            activation_ratio_provider=lambda config, revision: {1: 0.25},
        )

    assert backend.started is False


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
