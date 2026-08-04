"""SGLang subprocess backend with streamed OpenAI-compatible measurements."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import IO

import httpx

from moe_profiler.backends.base import (
    Backend,
    BackendError,
    BackendUnavailableError,
    GenerationResult,
)
from moe_profiler.workloads.base import RequestSpec


class SglangBackend(Backend):
    """Manage an SGLang server and measure streamed generation latency."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        startup_timeout_s: float = 600.0,
        request_timeout_s: float = 600.0,
    ) -> None:
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self._process: subprocess.Popen[str] | None = None
        self._server_log: IO[str] | None = None
        self._model_id: str | None = None

    @property
    def version(self) -> str:
        """Return package metadata without importing SGLang."""
        try:
            return importlib.metadata.version("sglang")
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    def start(
        self,
        model_id: str,
        quant: str | None = None,
        tp_size: int = 1,
        **kwargs: object,
    ) -> None:
        """Launch SGLang and poll its health endpoint.

        The future trace integration can pass
        ``expert_distribution_recorder_mode=...`` through ``kwargs``; it becomes
        SGLang's ``--expert-distribution-recorder-mode`` command-line flag.
        """
        if self._process is not None:
            raise BackendError("SGLang backend is already running")
        if importlib.util.find_spec("sglang") is None:
            raise BackendUnavailableError(
                "SGLang backend unavailable: the 'sglang' package is not installed "
                "in this environment. Install SGLang in a dedicated GPU environment."
            )

        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tp-size",
            str(tp_size),
        ]
        if quant:
            command.extend(["--quantization", quant])
        command.extend(self._format_extra_arguments(kwargs))

        popen_options: dict[str, object] = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True

        self._server_log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                text=True,
                **popen_options,
            )
            self._model_id = model_id
            self._wait_until_ready()
        except (OSError, BackendError) as exc:
            log_tail = self._read_log_tail()
            self.stop()
            if isinstance(exc, BackendError):
                detail = f"\nServer log:\n{log_tail}" if log_tail else ""
                raise type(exc)(f"{exc}{detail}") from exc
            raise BackendUnavailableError(f"Could not launch SGLang: {exc}") from exc

    def generate(self, requests: list[RequestSpec]) -> list[GenerationResult]:
        """Send requests concurrently and preserve their input ordering."""
        if self._process is None or self._process.poll() is not None:
            raise BackendError("SGLang backend is not running")
        if self._model_id is None:
            raise BackendError("SGLang backend has no configured model")
        if not requests:
            return []

        endpoint = f"http://{self.host}:{self.port}/v1/chat/completions"
        with httpx.Client(timeout=self.request_timeout_s) as client:
            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                return list(
                    executor.map(
                        lambda request: self._generate_one(client, endpoint, request),
                        requests,
                    )
                )

    def stop(self) -> None:
        """Terminate the owned server process group and close its log."""
        process = self._process
        self._process = None
        self._model_id = None
        if process is not None and process.poll() is None:
            self._terminate_process_tree(process)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._kill_process_tree(process)
                process.wait(timeout=5)
        if self._server_log is not None:
            self._server_log.close()
            self._server_log = None

    @staticmethod
    def _format_extra_arguments(kwargs: dict[str, object]) -> list[str]:
        arguments: list[str] = []
        for name, value in kwargs.items():
            if value is None or value is False:
                continue
            arguments.append(f"--{name.replace('_', '-')}")
            if value is not True:
                arguments.append(str(value))
        return arguments

    def _wait_until_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.startup_timeout_s
        health_url = f"http://{self.host}:{self.port}/health"
        with httpx.Client(timeout=2.0) as client:
            while time.monotonic() < deadline:
                exit_code = self._process.poll()
                if exit_code is not None:
                    raise BackendUnavailableError(
                        f"SGLang server exited during startup with code {exit_code}"
                    )
                try:
                    response = client.get(health_url)
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
        raise BackendUnavailableError(
            f"SGLang did not become healthy within {self.startup_timeout_s:.1f}s"
        )

    def _generate_one(
        self,
        client: httpx.Client,
        endpoint: str,
        request: RequestSpec,
    ) -> GenerationResult:
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started_at = time.perf_counter()
        first_chunk_at: float | None = None
        prompt_tokens = 0
        output_tokens = 0
        output_parts: list[str] = []

        try:
            with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    event = json.loads(data)
                    usage = event.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                    output_tokens = int(usage.get("completion_tokens", output_tokens))
                    choices = event.get("choices") or []
                    if choices:
                        content = (choices[0].get("delta") or {}).get("content")
                        if content:
                            output_parts.append(content)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise BackendError(
                f"Generation request {request.request_id!r} failed: {exc}"
            ) from exc

        finished_at = time.perf_counter()
        if first_chunk_at is None:
            raise BackendError(
                f"Generation request {request.request_id!r} returned no streamed chunks"
            )
        if output_tokens <= 0:
            raise BackendError(
                "The server did not report completion token usage; SGLang must support "
                "stream_options.include_usage to calculate TPOT."
            )
        ttft_s = first_chunk_at - started_at
        e2e_s = finished_at - started_at
        if output_tokens == 1:
            warnings.warn(
                f"Generation request {request.request_id!r} completed with one token; "
                "TPOT is undefined, so tpot_s is recorded as 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            tpot_s = 0.0
        else:
            tpot_s = (e2e_s - ttft_s) / (output_tokens - 1)
        return GenerationResult(
            request_id=request.request_id,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            e2e_s=e2e_s,
            output_text="".join(output_parts),
        )

    def _read_log_tail(self, max_characters: int = 4000) -> str:
        if self._server_log is None:
            return ""
        self._server_log.flush()
        self._server_log.seek(0)
        return self._server_log.read()[-max_characters:].strip()

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
