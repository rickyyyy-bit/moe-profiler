# moe-profiler

`moe-profiler` characterises sparse Mixture-of-Experts (MoE) inference while
keeping serving measurements, Hugging Face reference traces, analytical models,
synthetic demonstrations, and hardware counters distinct. Results carry enough
provenance to determine whether latency and routing observations describe the
same model revision, workload, phase, and trial.

The metric definitions follow the ideas in
[MoE-CAP](https://arxiv.org/abs/2412.07067). This repository is an independent,
small implementation intended for experiments and code review.

## Capabilities

This release provides:

- streamed TTFT/TPOT measurement through vLLM and SGLang;
- deterministic concurrency/sequence sweeps, warmups, repetitions, and raw logs;
- model statistics, analytical KV-cache growth, MBU/MFU, and sparse S-MBU/S-MFU;
- per-step Hugging Face expert reference tracing with coverage, load ratio,
  coefficient of variation, and normalized entropy;
- `torch.profiler` bandwidth extraction, validation, and roofline plots.

## Installation

Use Python 3.11 or newer for the core package and development checks:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

vLLM and SGLang can conflict, so install each server in its own virtual
environment rather than adding both to the core environment:

```bash
# vLLM environment
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -e .
python -m pip install vllm

# SGLang environment
python -m venv .venv-sglang
source .venv-sglang/bin/activate
python -m pip install -e .
python -m pip install "sglang[all]"
```

Backend and accelerator compatibility depends on the host platform. Supply
Hugging Face credentials only through `HF_TOKEN`. Trace mode defaults to
`device_map: auto`, so reinstall the declared `accelerate` dependency in an
older environment before tracing.

## Quickstart

Run offline analytical checks (no GPU or model download required):

```bash
ruff check .
black --check .
pytest
moe-profiler offline-check
python scripts/regenerate_figures.py
```

The last command reads the committed `results/demo_sweep.csv` and creates the
complete scoped figure manifest under `results/figures/`. The source CSV is an
explicitly synthetic demonstration dataset, not a hardware benchmark.

Create a Hugging Face reference trace. This is a separate execution from serving:

```bash
moe-profiler hf-trace --config configs/sweep_default.yaml
```

Run an uninstrumented serving sweep with the trace section disabled or set to
`mode: none`, or run a serving sweep plus a separately labelled reference trace:

```bash
moe-profiler sweep --config configs/sweep_default.yaml
```

On a supported GPU, smoke-test either serving backend:

```bash
python scripts/smoke_test.py --config configs/model_qwen15moe.yaml
python scripts/smoke_test.py --backend sglang --config configs/model_qwen15moe.yaml
```

Each sweep writes a summary CSV, raw per-trial JSONL, and raw per-request JSONL
under `results/`. Validate a result before reporting it with:

```bash
moe-profiler report results/<run-id>.csv
```

`moe-profiler backend-trace` currently exits with a clear unsupported message.
The vLLM and SGLang adapters do not simulate backend-native expert traces.

## Metrics

| Metric | Definition used here |
| --- | --- |
| TTFT | Client wall time immediately before the request to the first non-empty generated-token content; empty role/delta events do not stop the timer. |
| TPOT | `(end-to-end − TTFT) / (output tokens − 1)`; a one-token completion reports `0` with a warning because no inter-token interval exists. |
| Throughput | Output tokens divided by full batch wall time, including prefill; it is deliberately end-to-end throughput. |
| KV bytes/token | `2 × layers × KV heads × head dimension × dtype bytes`. |
| Activated ratio | Union of routed experts inside one reference forward pass, averaged across pinned workload manifests; shared experts are recorded separately. |
| MBU / S-MBU | Analytical weight/KV byte estimate per decode TPOT divided by peak bandwidth. S-MBU requires a matched backend-native decode trace unless exploratory HF-reference use is explicitly enabled. |
| MFU / S-MFU | Token throughput × total / active linear FLOPs per token divided by peak FLOPs. The attention term covers projections and excludes sequence-length-dependent score/value operations. |
| Arithmetic intensity | FLOPs divided by bytes moved; points left/right of the device ridge are memory/compute bound. |

All result rows include immutable model/tokenizer revisions, backend version,
dtype, tensor parallel size, concurrency, execution phase, trace mode, workload
fingerprint, requested/actual token counts, measurement kind, instrumentation,
and metric-availability reasons. Activation caches are compressed NPZ bundles
whose identities include every field that changes trace meaning; stale or
incompatible entries are not reused.

## Reproducible figures and findings

`scripts/regenerate_figures.py` owns every figure in the scoped report:

- `throughput_vs_batch.png` and `tpot_vs_batch.png` from the committed CSV;
- `activated_ratio_vs_batch.png` from per-batch demonstration ratios;
- `kv_vs_seqlen.png` from fixed Qwen2-MoE-like architecture dimensions;
- `roofline.png` from fixed illustrative H100 points and the device database.

The demonstration makes the expected systems relationships visible: throughput
initially improves with batching; latency eventually rises; expert activation
grows sub-linearly toward dense-equivalent coverage; KV memory grows linearly in
batch × sequence length; and decode has lower arithmetic intensity than prefill.
These are reproducibility examples, not measured performance claims. See
[`results/report.md`](results/report.md) for the scoped report and
[`results/notes.md`](results/notes.md) for the sparsity interpretation.

## Current validation status

| Capability | Implemented | Validated offline | Needs GPU/backend run |
| --- | --- | --- | --- |
| Provenance and workload fingerprints | Yes | Yes | No |
| Exact token-length manifests | Yes | Yes, with deterministic tokenizers | Real tokenizer/model access |
| HF reference trace and cache validation | Yes | Tiny CPU MoE | Trained-model trace |
| Per-step routing statistics | Yes | Tiny CPU MoE | Backend overhead measurement |
| Explicit sparse/shared/dense byte model | Yes | Hand-calculated fixtures | Hardware-counter comparison |
| Repeated serving benchmark and raw logs | Yes | Fake backend | vLLM/SGLang GPU run |
| Backend-native expert trace | No | Unsupported path tested | Adapter implementation and validation |
| Matched serving S-MBU | Compatibility gate only | Invalid combinations rejected | Backend-native decode trace |

The committed CSV, report figures, and roofline points are synthetic or
illustrative. GPU validation can use a smaller batch, another MoE model,
quantisation, CPU offload, tensor parallelism, or multiple devices; no particular
accelerator memory size is asserted as the only valid route.

## Module map

| Path | Responsibility |
| --- | --- |
| `backends/` | Backend ABC, process lifecycle, and streamed OpenAI-compatible clients. |
| `workloads/` | Typed request schema used by serving backends and sweeps. |
| `metrics/model_stats.py` | HF config normalisation, parameter counts, and token FLOPs. |
| `metrics/kv_cache.py` | Analytical KV-cache byte formulas. |
| `metrics/mbu_mfu.py` | Dense and sparsity-aware utilisation metrics. |
| `metrics/roofline.py` | Intensity, ridge, attainable roof, and bound classification. |
| `profiling/expert_recorder.py` | Router hooks, per-pass activation statistics, and cache files. |
| `profiling/torch_prof.py` | DRAM-traffic extraction and analytical/profiler validation. |
| `runner/` | Sweep orchestration and stable `RunResult` CSV schema. |
| `cost/devices.py` | Explicit accelerator peak-bandwidth/FLOP database. |
| `viz/` | Batch, sparsity, KV, and roofline plotting code. |
| `scripts/` | Smoke runner and one-command figure regeneration. |

## Limitations

- The committed CSV and roofline points are synthetic/illustrative; replace them
  with real revision-stamped GPU runs before making performance claims.
- Tiny/random MoE fixtures are suitable only for hook integration tests. Their
  random routers and small expert pools can saturate at batch 1, so their
  activation ratios must not be reported as model findings.
- Hugging Face hooks produce a reference trace, not a trace of vLLM or SGLang.
- Expert-module discovery uses architecture heuristics and should be checked when
  adding a new model family.
- Profiler kernels do not always expose DRAM-byte counters. The code reports the
  counter source/fallback and should not turn an estimate into a measurement.
- Device peaks and prices are approximate reference values, not live quotes.
