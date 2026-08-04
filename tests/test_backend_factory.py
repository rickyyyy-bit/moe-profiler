"""Tests for backend selection and engine-agnostic smoke orchestration."""

from unittest.mock import patch

import pytest
from scripts.smoke_test import run

from moe_profiler.backends import get_backend
from moe_profiler.backends.base import Backend, GenerationResult
from moe_profiler.backends.sglang_backend import SglangBackend
from moe_profiler.backends.vllm_backend import VllmBackend
from moe_profiler.config import ModelConfig
from moe_profiler.workloads.base import RequestSpec


@pytest.mark.parametrize(
    ("name", "backend_type"),
    [("vllm", VllmBackend), ("SGLANG", SglangBackend)],
)
def test_get_backend_returns_requested_implementation(
    name: str, backend_type: type[Backend]
) -> None:
    backend = get_backend(name, port=31000)

    assert isinstance(backend, backend_type)
    assert backend.port == 31000


def test_get_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        get_backend("unknown")


def test_smoke_runner_is_polymorphic(capsys: pytest.CaptureFixture[str]) -> None:
    backend = _FakeBackend()
    config = ModelConfig(model_id="test/model", dtype="bfloat16", tp_size=2)

    with patch("scripts.smoke_test.get_backend", return_value=backend) as factory:
        run(config, "sglang")

    factory.assert_called_once_with(
        "sglang",
        host=config.host,
        port=config.port,
        startup_timeout_s=config.startup_timeout_s,
    )
    assert backend.start_arguments == (
        "test/model",
        None,
        2,
        {"dtype": "bfloat16"},
    )
    assert backend.stopped is True
    output = capsys.readouterr().out
    assert "ttft_s=0.100000" in output
    assert "tpot_s=0.050000" in output
    assert "throughput_tok_s=10.000000" in output


class _FakeBackend(Backend):
    def __init__(self) -> None:
        self.start_arguments: tuple[str, str | None, int, dict[str, object]] | None = (
            None
        )
        self.stopped = False

    def start(
        self,
        model_id: str,
        quant: str | None = None,
        tp_size: int = 1,
        **kwargs: object,
    ) -> None:
        self.start_arguments = (model_id, quant, tp_size, kwargs)

    def generate(self, requests: list[RequestSpec]) -> list[GenerationResult]:
        return [
            GenerationResult(
                request_id=requests[0].request_id,
                prompt_tokens=4,
                output_tokens=5,
                ttft_s=0.1,
                tpot_s=0.05,
                e2e_s=0.5,
                output_text="test output",
            )
        ]

    def stop(self) -> None:
        self.stopped = True

    @property
    def version(self) -> str:
        return "test"
