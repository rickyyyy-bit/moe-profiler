"""Unit tests for streamed latency parsing without requiring vLLM."""

import json
from unittest.mock import patch

import httpx
import pytest

from moe_profiler.backends.base import BackendUnavailableError
from moe_profiler.backends.vllm_backend import VllmBackend
from moe_profiler.workloads.base import RequestSpec


def _streaming_response(request: httpx.Request) -> httpx.Response:
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Sparse"}}]},
        {"choices": [{"delta": {"content": " output"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    ]

    body = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    return httpx.Response(200, content=body + b"data: [DONE]\n\n")


def test_generate_parses_usage_and_streamed_text() -> None:
    backend = VllmBackend()
    backend._process = _RunningProcess()  # type: ignore[assignment]
    backend._model_id = "test/model"
    request = RequestSpec(request_id="request-1", prompt="hello", max_tokens=8)

    transport = httpx.MockTransport(_streaming_response)
    with patch("httpx.Client", return_value=httpx.Client(transport=transport)):
        result = backend.generate([request])[0]

    assert result.request_id == "request-1"
    assert result.prompt_tokens == 5
    assert result.output_tokens == 2
    assert result.output_text == "Sparse output"
    assert 0 <= result.ttft_s <= result.e2e_s
    assert result.tpot_s == pytest.approx(result.e2e_s - result.ttft_s)


def test_start_reports_missing_vllm_cleanly() -> None:
    backend = VllmBackend()
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(BackendUnavailableError, match="command is not installed"),
    ):
        backend.start("test/model")


class _RunningProcess:
    def poll(self) -> None:
        return None
