"""Unit tests for SGLang without importing or launching SGLang."""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from moe_profiler.backends.base import BackendUnavailableError
from moe_profiler.backends.sglang_backend import SglangBackend
from moe_profiler.workloads.base import RequestSpec


def _streaming_response(request: httpx.Request) -> httpx.Response:
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "SGLang"}}]},
        {"choices": [{"delta": {"content": " output"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    ]
    body = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    return httpx.Response(200, content=body + b"data: [DONE]\n\n")


def test_start_builds_sglang_launch_command() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.poll.return_value = 0
    backend = SglangBackend(host="127.0.0.1", port=31000)

    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("subprocess.Popen", return_value=process) as popen,
        patch.object(backend, "_wait_until_ready"),
    ):
        backend.start(
            "test/model",
            tp_size=2,
            expert_distribution_recorder_mode="stat",
        )

    command = popen.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "sglang.launch_server"]
    assert command[command.index("--model-path") + 1] == "test/model"
    assert command[command.index("--port") + 1] == "31000"
    assert command[command.index("--tp-size") + 1] == "2"
    assert command[command.index("--expert-distribution-recorder-mode") + 1] == "stat"
    backend.stop()


def test_generate_matches_backend_result_contract() -> None:
    backend = SglangBackend()
    backend._process = _RunningProcess()  # type: ignore[assignment]
    backend._model_id = "test/model"
    request = RequestSpec(request_id="request-1", prompt="hello", max_tokens=8)
    transport = httpx.MockTransport(_streaming_response)

    with patch("httpx.Client", return_value=httpx.Client(transport=transport)):
        result = backend.generate([request])[0]

    assert result.request_id == "request-1"
    assert result.prompt_tokens == 4
    assert result.output_tokens == 2
    assert result.output_text == "SGLang output"
    assert 0 <= result.ttft_s <= result.e2e_s
    assert result.tpot_s == pytest.approx(result.e2e_s - result.ttft_s)


def test_generate_records_zero_tpot_for_single_token() -> None:
    def single_token_response(request: httpx.Request) -> httpx.Response:
        del request
        events = [
            {"choices": [{"delta": {"content": "Done"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        ]
        body = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
        return httpx.Response(200, content=body + b"data: [DONE]\n\n")

    backend = SglangBackend()
    backend._process = _RunningProcess()  # type: ignore[assignment]
    backend._model_id = "test/model"
    request = RequestSpec(request_id="request-1", prompt="hello", max_tokens=1)
    transport = httpx.MockTransport(single_token_response)

    with (
        patch("httpx.Client", return_value=httpx.Client(transport=transport)),
        pytest.warns(RuntimeWarning, match="TPOT is undefined"),
    ):
        result = backend.generate([request])[0]

    assert result.output_tokens == 1
    assert result.tpot_s == 0.0
    assert result.output_text == "Done"


def test_start_reports_missing_sglang_cleanly() -> None:
    backend = SglangBackend()
    with (
        patch("importlib.util.find_spec", return_value=None),
        pytest.raises(BackendUnavailableError, match="package is not installed"),
    ):
        backend.start("test/model")


class _RunningProcess:
    def poll(self) -> None:
        return None
