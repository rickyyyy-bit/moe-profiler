"""Ensure concrete backends issue a request list concurrently."""

from threading import Barrier
from unittest.mock import patch

import pytest

from moe_profiler.backends.base import GenerationResult
from moe_profiler.backends.sglang_backend import SglangBackend
from moe_profiler.backends.vllm_backend import VllmBackend
from moe_profiler.workloads.base import RequestSpec


@pytest.mark.parametrize("backend", [VllmBackend(), SglangBackend()])
def test_generate_runs_batch_concurrently_and_preserves_order(
    backend: VllmBackend | SglangBackend,
) -> None:
    backend._process = _RunningProcess()  # type: ignore[assignment]
    backend._model_id = "test/model"
    requests = [
        RequestSpec(request_id=f"request-{index}", prompt="hello", max_tokens=8)
        for index in range(3)
    ]
    barrier = Barrier(len(requests))

    def generate_one(
        client: object, endpoint: str, request: RequestSpec
    ) -> GenerationResult:
        del client, endpoint
        barrier.wait(timeout=2.0)
        return GenerationResult(
            request_id=request.request_id,
            prompt_tokens=1,
            output_tokens=2,
            ttft_s=0.01,
            tpot_s=0.01,
            e2e_s=0.02,
            output_text="ok",
        )

    with patch.object(backend, "_generate_one", side_effect=generate_one):
        results = backend.generate(requests)

    assert [result.request_id for result in results] == [
        request.request_id for request in requests
    ]


class _RunningProcess:
    def poll(self) -> None:
        return None
