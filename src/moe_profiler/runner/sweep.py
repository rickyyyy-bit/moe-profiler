"""Controlled batch-size and sequence-length sweep orchestration."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from moe_profiler.backends import get_backend
from moe_profiler.backends.base import Backend, GenerationResult
from moe_profiler.config import AppConfig, SweepConfig
from moe_profiler.metrics.assembly import AssembledMetrics, assemble_metrics
from moe_profiler.metrics.kv_cache import kv_cache_bytes
from moe_profiler.metrics.model_stats import ModelStats, load_model_stats
from moe_profiler.provenance import MeasurementProvenance, TraceCacheIdentity
from moe_profiler.runner.record import ResultWriter, RunResult
from moe_profiler.workloads.base import WorkloadManifest, build_exact_length_manifest


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
    activation_ratio_provider: (
        Callable[[AppConfig, str], Mapping[int, float]] | None
    ) = None,
    tokenizer: object | None = None,
    stats_loader: Callable[[str, float, str], ModelStats] | None = None,
) -> Path:
    """Run warmups and repetitions, preserving raw trials and summary rows."""
    app_config = load_sweep_config(config) if isinstance(config, str | Path) else config
    if app_config.sweep is None:
        raise ValueError("AppConfig.sweep is required")
    sweep = app_config.sweep
    random.seed(sweep.seed)
    resolver = revision_resolver or resolve_model_revision
    model_revision = resolver(
        app_config.model.model_id, app_config.model.model_revision
    )
    if tokenizer is None:
        tokenizer = (
            _OfflineWhitespaceTokenizer()
            if backend is not None
            else _load_tokenizer(app_config, model_revision)
        )
    manifests = _build_manifests(app_config, model_revision, tokenizer, run_id=None)
    activation_ratios = _resolve_activation_ratios(
        app_config,
        model_revision,
        provider=activation_ratio_provider,
        manifests=manifests,
    )
    run_id = uuid.uuid4().hex
    manifests = _build_manifests(app_config, model_revision, tokenizer, run_id=run_id)
    writer = ResultWriter(run_id, results_dir=sweep.results_dir)
    sweep.results_dir.mkdir(parents=True, exist_ok=True)
    trial_path = sweep.results_dir / f"{run_id}.trials.jsonl"
    request_path = sweep.results_dir / f"{run_id}.requests.jsonl"
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
    stats = _load_stats_if_requested(app_config, model_revision, stats_loader)
    environment_json = json.dumps(_environment_metadata(), sort_keys=True)

    try:
        serving_backend.start(
            app_config.model.model_id,
            quant=app_config.model.quant,
            tp_size=app_config.model.tp_size,
            **start_options,
        )
        with (
            trial_path.open("w", encoding="utf-8") as trial_file,
            request_path.open("w", encoding="utf-8") as request_file,
        ):
            for input_len, output_len in sweep.sequence_lengths:
                for batch_size in sweep.batch_sizes:
                    manifest = manifests[(input_len, output_len, batch_size)]
                    requests = manifest.request_specs()
                    for warmup_index in range(sweep.warmup_trials):
                        warmup_id = f"warmup-{warmup_index}"
                        warmup_generations = serving_backend.generate(requests)
                        if len(warmup_generations) != batch_size:
                            raise ValueError(
                                "backend returned an unexpected number of warmup "
                                "results"
                            )
                        warmup_generations = _attach_provenance(
                            warmup_generations,
                            manifest=manifest,
                            run_id=run_id,
                            trial_id=warmup_id,
                            model_id=app_config.model.model_id,
                            model_revision=model_revision,
                            backend_name=sweep.backend,
                            backend_version=serving_backend.version,
                            dtype=app_config.model.dtype,
                            quantization=app_config.model.quant,
                            tp_size=app_config.model.tp_size,
                            concurrency=batch_size,
                            warmup=True,
                        )
                        trial_file.write(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "trial_id": warmup_id,
                                    "warmup": True,
                                    "workload_fingerprint": manifest.fingerprint,
                                    "concurrency": batch_size,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        for generation in warmup_generations:
                            request_file.write(
                                json.dumps(
                                    _raw_request_record(
                                        generation,
                                        run_id=run_id,
                                        trial_id=warmup_id,
                                        workload_fingerprint=manifest.fingerprint,
                                    ),
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                    all_generations: list[GenerationResult] = []
                    elapsed_trials: list[float] = []
                    for repetition in range(sweep.measured_repetitions):
                        trial_id = f"trial-{repetition}"
                        started_at = clock()
                        generations = serving_backend.generate(requests)
                        elapsed_s = clock() - started_at
                        _validate_generations(generations, batch_size, elapsed_s)
                        generations = _attach_provenance(
                            generations,
                            manifest=manifest,
                            run_id=run_id,
                            trial_id=trial_id,
                            model_id=app_config.model.model_id,
                            model_revision=model_revision,
                            backend_name=sweep.backend,
                            backend_version=serving_backend.version,
                            dtype=app_config.model.dtype,
                            quantization=app_config.model.quant,
                            tp_size=app_config.model.tp_size,
                            concurrency=batch_size,
                        )
                        all_generations.extend(generations)
                        elapsed_trials.append(elapsed_s)
                        trial_file.write(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "trial_id": trial_id,
                                    "warmup": False,
                                    "workload_fingerprint": manifest.fingerprint,
                                    "concurrency": batch_size,
                                    "elapsed_s": elapsed_s,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        for generation in generations:
                            request_file.write(
                                json.dumps(
                                    _raw_request_record(
                                        generation,
                                        run_id=run_id,
                                        trial_id=trial_id,
                                        workload_fingerprint=manifest.fingerprint,
                                    ),
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                    actual_prompt = round(
                        statistics.fmean(item.prompt_tokens for item in all_generations)
                    )
                    actual_output = round(
                        statistics.fmean(item.output_tokens for item in all_generations)
                    )
                    mean_tpot = statistics.fmean(
                        item.tpot_s for item in all_generations
                    )
                    throughput = statistics.fmean(
                        sum(
                            item.output_tokens
                            for item in all_generations[
                                repetition * batch_size : (repetition + 1) * batch_size
                            ]
                        )
                        / elapsed_trials[repetition]
                        for repetition in range(sweep.measured_repetitions)
                    )
                    kv_bytes = (
                        None
                        if stats is None
                        else float(
                            kv_cache_bytes(
                                stats,
                                seq_len=actual_prompt + actual_output,
                                batch=batch_size,
                            )
                        )
                    )
                    metrics = assemble_metrics(
                        stats=stats,
                        tpot_s=mean_tpot,
                        throughput_tok_s=throughput,
                        kv_bytes=kv_bytes,
                        peak_bw_bytes_s=(
                            None
                            if sweep.peak_bw_gbps is None
                            else sweep.peak_bw_gbps * 1e9
                        ),
                        peak_flops=sweep.peak_flops,
                        trace_mode=(
                            "none"
                            if app_config.trace is None
                            else app_config.trace.mode
                        ),
                    )
                    result = _aggregate_result(
                        generations=all_generations,
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
                        elapsed_s=statistics.fmean(elapsed_trials),
                        activated_param_ratio=activation_ratios.get(batch_size),
                        repetitions=sweep.measured_repetitions,
                        manifest=manifest,
                        dtype=app_config.model.dtype,
                        tp_size=app_config.model.tp_size,
                        trace_mode=(
                            "none"
                            if app_config.trace is None
                            else app_config.trace.mode
                        ),
                        metrics=metrics,
                        environment_json=environment_json,
                    )
                    writer.append(result)
    finally:
        serving_backend.stop()
    return writer.path


def measure_activation_ratios(
    config: AppConfig,
    model_revision: str,
    manifests: Mapping[tuple[int, int, int], WorkloadManifest] | None = None,
) -> dict[int, float]:
    """Trace the same pinned manifests as an explicitly separate HF execution."""
    if config.sweep is None or config.trace is None:
        raise ValueError("sweep and trace configurations are required")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "trace sweeps require torch and transformers; install project dependencies"
        ) from exc

    from moe_profiler.profiling.expert_recorder import ExpertActivationRecorder

    trace = config.trace
    model_options: dict[str, object] = {
        "revision": model_revision,
        "trust_remote_code": trace.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    dtype = _torch_dtype(config.model.dtype, torch)
    if dtype is not None:
        model_options["torch_dtype"] = dtype
    if trace.device_map is not None:
        model_options["device_map"] = trace.device_map

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=model_revision,
        trust_remote_code=trace.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("trace tokenizer defines neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = None
    recorder = ExpertActivationRecorder(cache_dir=trace.cache_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            **model_options,
        )
        recorder.attach(model)
        if manifests is None:
            manifests = _build_manifests(
                config, model_revision, tokenizer, run_id="hf-reference"
            )
        ratios_by_concurrency: dict[int, list[float]] = {}
        for (_input_len, _output_len, batch_size), manifest in manifests.items():
            identity = TraceCacheIdentity(
                model_id=config.model.model_id,
                model_revision=model_revision,
                tokenizer_id=manifest.tokenizer_id,
                tokenizer_revision=manifest.tokenizer_revision,
                dtype=config.model.dtype,
                workload_fingerprint=manifest.fingerprint,
                trace_mode="hf_reference",
                phase="prefill",
                prompt_token_lengths=tuple(
                    request.actual_prompt_tokens for request in manifest.requests
                ),
                output_token_lengths=tuple(
                    request.max_output_tokens for request in manifest.requests
                ),
                concurrency=batch_size,
                top_k=recorder.top_k or 1,
                tracer_version=trace.tracer_version,
            )
            recorder.run(
                model,
                tokenizer,
                [request.prompt for request in manifest.requests],
                1,
                batch_size=batch_size,
                cache_identity=identity,
                phase="prefill",
            )
            ratios_by_concurrency.setdefault(batch_size, []).append(
                recorder.activated_ratio(batch_size)
            )
        return {
            concurrency: statistics.fmean(values)
            for concurrency, values in ratios_by_concurrency.items()
        }
    finally:
        recorder.detach()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def _load_tokenizer(config: AppConfig, model_revision: str) -> object:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to build verified workloads"
        ) from exc
    return AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=model_revision,
        trust_remote_code=(
            False if config.trace is None else config.trace.trust_remote_code
        ),
    )


def _build_manifests(
    config: AppConfig,
    model_revision: str,
    tokenizer: object,
    *,
    run_id: str | None,
) -> dict[tuple[int, int, int], WorkloadManifest]:
    del run_id  # Manifest identity must remain stable across benchmark runs.
    if config.sweep is None:
        raise ValueError("AppConfig.sweep is required")
    manifests: dict[tuple[int, int, int], WorkloadManifest] = {}
    for input_len, output_len in config.sweep.sequence_lengths:
        for concurrency in config.sweep.batch_sizes:
            key = (input_len, output_len, concurrency)
            manifests[key] = build_exact_length_manifest(
                tokenizer,
                tokenizer_id=config.model.model_id,
                tokenizer_revision=model_revision,
                request_prefix=(
                    f"wl-in{input_len}-out{output_len}-concurrency{concurrency}"
                ),
                concurrency=concurrency,
                prompt_tokens=input_len,
                output_tokens=output_len,
            )
    return manifests


def _load_stats_if_requested(
    config: AppConfig,
    model_revision: str,
    loader: Callable[[str, float, str], ModelStats] | None,
) -> ModelStats | None:
    if config.sweep is None:
        return None
    if config.sweep.peak_bw_gbps is None and config.sweep.peak_flops is None:
        return None
    dtype_bytes = _dtype_bytes(config.model.dtype, config.model.quant)
    if loader is not None:
        return loader(config.model.model_id, dtype_bytes, model_revision)
    return load_model_stats(
        config.model.model_id,
        dtype_bytes,
        revision=model_revision,
    )


def _dtype_bytes(dtype: str, quant: str | None) -> float:
    normalized_quant = "" if quant is None else quant.lower()
    if "4" in normalized_quant:
        return 0.5
    if "8" in normalized_quant:
        return 1.0
    normalized = dtype.lower()
    if normalized in {"float32", "fp32"}:
        return 4.0
    if normalized in {"float16", "fp16", "bfloat16", "bf16", "auto"}:
        return 2.0
    raise ValueError(f"cannot infer byte width for dtype {dtype!r}")


def _validate_generations(
    generations: list[GenerationResult], concurrency: int, elapsed_s: float
) -> None:
    if len(generations) != concurrency:
        raise ValueError(
            f"backend returned {len(generations)} results for concurrency {concurrency}"
        )
    if elapsed_s <= 0:
        raise ValueError("measured sweep duration must be positive")


def _attach_provenance(
    generations: list[GenerationResult],
    *,
    manifest: WorkloadManifest,
    run_id: str,
    trial_id: str,
    model_id: str,
    model_revision: str,
    backend_name: str,
    backend_version: str,
    dtype: str,
    quantization: str | None,
    tp_size: int,
    concurrency: int,
    warmup: bool = False,
) -> list[GenerationResult]:
    requests = {request.request_id: request for request in manifest.requests}
    enriched: list[GenerationResult] = []
    for generation in generations:
        try:
            request = requests[generation.request_id]
        except KeyError as exc:
            raise ValueError(
                f"backend returned unknown request ID {generation.request_id!r}"
            ) from exc
        provenance = MeasurementProvenance(
            run_id=run_id,
            trial_id=trial_id,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=manifest.tokenizer_id,
            tokenizer_revision=manifest.tokenizer_revision,
            backend_name=backend_name,
            backend_version=backend_version,
            dtype=dtype,
            quantization=quantization,
            tensor_parallel_size=tp_size,
            workload_fingerprint=manifest.fingerprint,
            requested_prompt_tokens=request.requested_prompt_tokens,
            actual_prompt_tokens=generation.prompt_tokens,
            requested_output_tokens=request.max_output_tokens,
            actual_output_tokens=generation.output_tokens,
            concurrency=concurrency,
            batch_size=None,
            execution_phase="end_to_end",
            warmup=warmup,
            value_kind="measurement",
            instrumentation_mode="none",
            instrumentation_version="none",
        )
        enriched.append(generation.model_copy(update={"provenance": provenance}))
    return enriched


def _raw_request_record(
    generation: GenerationResult,
    *,
    run_id: str,
    trial_id: str,
    workload_fingerprint: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "trial_id": trial_id,
        "request_id": generation.request_id,
        "prompt_tokens": generation.prompt_tokens,
        "output_tokens": generation.output_tokens,
        "ttft_s": generation.ttft_s,
        "tpot_s": generation.tpot_s,
        "e2e_s": generation.e2e_s,
        "output_text": generation.output_text,
        "workload_fingerprint": workload_fingerprint,
        "provenance": (
            None
            if generation.provenance is None
            else generation.provenance.model_dump(mode="json")
        ),
    }


def _environment_metadata() -> dict[str, object]:
    packages: dict[str, str] = {}
    for package in ("torch", "transformers", "vllm", "sglang"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    metadata: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "driver": "unavailable",
    }
    try:
        import torch

        metadata["cuda"] = torch.version.cuda
        metadata["device"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "unavailable"
        )
    except ImportError:
        metadata["cuda"] = "unavailable"
        metadata["device"] = "unavailable"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        versions = sorted(set(completed.stdout.split()))
        if completed.returncode == 0 and versions:
            metadata["driver"] = versions
    return metadata


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class _OfflineWhitespaceTokenizer:
    """Deterministic test tokenizer used only with an injected offline backend."""

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise ValueError("offline tokenizer does not add special tokens")
        return list(range(1, len(text.split()) + 1))

    @staticmethod
    def decode(token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(f"t{token_id}" for token_id in token_ids)


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
    activated_param_ratio: float | None,
    repetitions: int = 1,
    manifest: WorkloadManifest | None = None,
    dtype: str = "unknown",
    tp_size: int = 1,
    trace_mode: str = "none",
    metrics: AssembledMetrics | None = None,
    environment_json: str = "{}",
) -> RunResult:
    if len(generations) != batch_size * repetitions:
        raise ValueError(
            f"backend returned {len(generations)} results for concurrency "
            f"{batch_size} across {repetitions} repetitions"
        )
    if elapsed_s <= 0:
        raise ValueError("measured sweep duration must be positive")
    total_output_tokens = sum(result.output_tokens for result in generations)
    total_prompt_tokens = sum(result.prompt_tokens for result in generations)
    max_ttft_s = max(result.ttft_s for result in generations)
    prefill_throughput = (
        total_prompt_tokens / repetitions / max_ttft_s if max_ttft_s > 0 else 0.0
    )
    # Deliberately end-to-end: elapsed_s includes prefill/TTFT and decode time.
    end_to_end_output_throughput = total_output_tokens / repetitions / elapsed_s
    e2e_values = [result.e2e_s for result in generations]
    status = {} if metrics is None else dict(metrics.status)
    if trace_mode == "hf_reference" and activated_param_ratio is not None:
        status["activated_param_ratio"] = (
            "hf_reference: separate execution; not a matched serving trace"
        )
    requested_prompt = input_len
    fingerprint = "unknown"
    tokenizer_id = None
    tokenizer_revision = None
    if manifest is not None:
        requested_prompt = manifest.requests[0].requested_prompt_tokens
        fingerprint = manifest.fingerprint
        tokenizer_id = manifest.tokenizer_id
        tokenizer_revision = manifest.tokenizer_revision
    return RunResult(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        model=model,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        backend=backend_name,
        backend_version=backend_version,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        quant=quant,
        device=sweep.device,
        workload=sweep.workload,
        workload_fingerprint=fingerprint,
        batch_size=batch_size,
        concurrency=batch_size,
        input_len=input_len,
        output_len=output_len,
        requested_prompt_tokens=requested_prompt,
        actual_prompt_tokens=round(total_prompt_tokens / len(generations)),
        requested_output_tokens=output_len,
        actual_output_tokens=round(total_output_tokens / len(generations)),
        trace_mode=trace_mode,
        instrumentation_mode="none",
        instrumentation_version="none",
        ttft_s=statistics.fmean(result.ttft_s for result in generations),
        tpot_s=statistics.fmean(result.tpot_s for result in generations),
        e2e_s=statistics.fmean(result.e2e_s for result in generations),
        throughput_tok_s=end_to_end_output_throughput,
        prefill_throughput_tok_s=prefill_throughput,
        activated_param_ratio=activated_param_ratio,
        vanilla_mbu=(None if metrics is None else metrics.vanilla_mbu),
        s_mbu=(None if metrics is None else metrics.s_mbu),
        vanilla_mfu=(None if metrics is None else metrics.vanilla_mfu),
        s_mfu=(None if metrics is None else metrics.s_mfu),
        latency_p50_s=_percentile(e2e_values, 50),
        latency_p95_s=_percentile(e2e_values, 95),
        latency_p99_s=_percentile(e2e_values, 99),
        metric_status_json=json.dumps(status, sort_keys=True),
        environment_json=environment_json,
    )


def _resolve_activation_ratios(
    config: AppConfig,
    model_revision: str,
    *,
    provider: Callable[[AppConfig, str], Mapping[int, float]] | None,
    manifests: Mapping[tuple[int, int, int], WorkloadManifest],
) -> dict[int, float]:
    if config.sweep is None:
        raise ValueError("AppConfig.sweep is required")
    if config.trace is None or not config.trace.enabled:
        return {}
    if config.trace.mode == "none":
        return {}
    if config.trace.mode == "backend_native":
        raise RuntimeError(
            "backend-native traces are unsupported by the current vLLM and "
            "SGLang adapters; no synthetic trace will be substituted"
        )
    ratios = dict(
        measure_activation_ratios(config, model_revision, manifests)
        if provider is None
        else provider(config, model_revision)
    )
    expected = set(config.sweep.batch_sizes)
    missing = expected - set(ratios)
    if missing:
        raise ValueError(
            f"activation ratio provider omitted batch sizes: {sorted(missing)}"
        )
    for batch_size in expected:
        ratio = ratios[batch_size]
        if isinstance(ratio, bool) or not isinstance(ratio, int | float):
            raise TypeError(f"activation ratio for batch {batch_size} must be numeric")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"activation ratio for batch {batch_size} must be between 0 and 1"
            )
    return {batch_size: float(ratios[batch_size]) for batch_size in expected}


def _torch_dtype(dtype_name: str, torch: object) -> object | None:
    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        return None
    aliases = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "float32": "float32",
        "fp32": "float32",
    }
    try:
        attribute = aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported trace dtype {dtype_name!r}") from exc
    return getattr(torch, attribute)
