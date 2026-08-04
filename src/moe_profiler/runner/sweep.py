"""Controlled batch-size and sequence-length sweep orchestration."""

from __future__ import annotations

import random
import statistics
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from moe_profiler.backends import get_backend
from moe_profiler.backends.base import Backend, GenerationResult
from moe_profiler.config import AppConfig, SweepConfig
from moe_profiler.runner.record import ResultWriter, RunResult
from moe_profiler.workloads.base import RequestSpec


def load_sweep_config(path: str | Path) -> AppConfig:
    """Load a nested model/sweep YAML configuration."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read sweep config {config_path}: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise ValueError(f"sweep config {config_path} must contain a YAML mapping")
    try:
        config = AppConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"invalid sweep config {config_path}: {exc}") from exc
    if config.sweep is None:
        raise ValueError(f"sweep config {config_path} must define a sweep section")
    return config


def run_sweep(
    config: AppConfig | str | Path,
    *,
    backend: Backend | None = None,
    clock: Callable[[], float] = time.perf_counter,
    revision_resolver: Callable[[str, str], str] | None = None,
) -> Path:
    """Run every batch/sequence point and write one aggregate row per point."""
    app_config = load_sweep_config(config) if isinstance(config, str | Path) else config
    if app_config.sweep is None:
        raise ValueError("AppConfig.sweep is required")
    sweep = app_config.sweep
    random.seed(sweep.seed)
    resolver = revision_resolver or resolve_model_revision
    model_revision = resolver(
        app_config.model.model_id, app_config.model.model_revision
    )
    run_id = uuid.uuid4().hex
    writer = ResultWriter(run_id, results_dir=sweep.results_dir)
    serving_backend = backend or get_backend(
        sweep.backend,
        host=app_config.model.host,
        port=app_config.model.port,
        startup_timeout_s=app_config.model.startup_timeout_s,
    )

    start_options: dict[str, object] = {
        "dtype": app_config.model.dtype,
        "revision": model_revision,
    }

    try:
        serving_backend.start(
            app_config.model.model_id,
            quant=app_config.model.quant,
            tp_size=app_config.model.tp_size,
            **start_options,
        )
        for input_len, output_len in sweep.sequence_lengths:
            for batch_size in sweep.batch_sizes:
                requests = _build_requests(
                    run_id=run_id,
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                )
                started_at = clock()
                generations = serving_backend.generate(requests)
                elapsed_s = clock() - started_at
                result = _aggregate_result(
                    generations=generations,
                    run_id=run_id,
                    model=app_config.model.model_id,
                    model_revision=model_revision,
                    backend_name=sweep.backend,
                    backend_version=serving_backend.version,
                    quant=app_config.model.quant,
                    sweep=sweep,
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                    elapsed_s=elapsed_s,
                )
                writer.append(result)
    finally:
        serving_backend.stop()
    return writer.path


def resolve_model_revision(model_id: str, requested_revision: str) -> str:
    """Resolve a branch, tag, or ``auto`` to an immutable HF commit SHA."""
    normalized = requested_revision.strip()
    if len(normalized) == 40 and all(
        character in "0123456789abcdefABCDEF" for character in normalized
    ):
        return normalized.lower()
    try:
        from huggingface_hub import model_info

        revision = None if normalized in {"auto", "unknown"} else normalized
        resolved = model_info(model_id, revision=revision).sha
    except Exception as exc:
        raise RuntimeError(
            f"could not resolve an immutable revision for {model_id!r}: {exc}"
        ) from exc
    if not resolved:
        raise RuntimeError(f"Hugging Face returned no revision SHA for {model_id!r}")
    return resolved


def _build_requests(
    *,
    run_id: str,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> list[RequestSpec]:
    prompt = " ".join(["token"] * input_len)
    return [
        RequestSpec(
            request_id=(
                f"{run_id}-bs{batch_size}-in{input_len}-out{output_len}-req{index}"
            ),
            prompt=prompt,
            max_tokens=output_len,
            metadata={"target_input_len": input_len},
        )
        for index in range(batch_size)
    ]


def _aggregate_result(
    *,
    generations: list[GenerationResult],
    run_id: str,
    model: str,
    model_revision: str,
    backend_name: str,
    backend_version: str,
    quant: str | None,
    sweep: SweepConfig,
    batch_size: int,
    input_len: int,
    output_len: int,
    elapsed_s: float,
) -> RunResult:
    if len(generations) != batch_size:
        raise ValueError(
            f"backend returned {len(generations)} results for batch size {batch_size}"
        )
    if elapsed_s <= 0:
        raise ValueError("measured sweep duration must be positive")
    total_output_tokens = sum(result.output_tokens for result in generations)
    total_prompt_tokens = sum(result.prompt_tokens for result in generations)
    max_ttft_s = max(result.ttft_s for result in generations)
    prefill_throughput = total_prompt_tokens / max_ttft_s if max_ttft_s > 0 else 0.0
    return RunResult(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        model=model,
        model_revision=model_revision,
        backend=backend_name,
        backend_version=backend_version,
        quant=quant,
        device=sweep.device,
        workload=sweep.workload,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        ttft_s=statistics.fmean(result.ttft_s for result in generations),
        tpot_s=statistics.fmean(result.tpot_s for result in generations),
        e2e_s=statistics.fmean(result.e2e_s for result in generations),
        throughput_tok_s=total_output_tokens / elapsed_s,
        prefill_throughput_tok_s=prefill_throughput,
    )
