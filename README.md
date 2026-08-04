# moe-profiler

`moe-profiler` characterises sparse Mixture-of-Experts (MoE) inference without
mistaking total model size for the parameters used by each token. It combines
serving measurements, Hugging Face router traces, analytical memory/FLOP models,
and profiler validation to explain where sparsity helps—and where growing batches
activate enough experts to erase that advantage.

The metric definitions follow the ideas in
[MoE-CAP](https://arxiv.org/abs/2412.07067). This repository is an independent,
small implementation intended for experiments and code review.

## Implemented scope

This release covers Stages 0–8 and the Stage 13 reproducibility/report layer:

- streamed TTFT/TPOT measurement through vLLM and SGLang;
- deterministic batch/sequence sweeps and result logging;
- model statistics, analytical KV-cache growth, MBU/MFU, and sparse S-MBU/S-MFU;
- per-forward-pass expert activation tracing and sparsity-vs-batch plots;
- `torch.profiler` bandwidth extraction, validation, and roofline plots.

Stages 9–12 (quantisation/accuracy, CAP radar, and agentic workloads) were skipped
by project decision. No CAP or agentic results are claimed in this release.

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

Run all offline checks (no GPU or serving engine required):

```bash
ruff check .
black --check .
pytest
python scripts/regenerate_figures.py
```

The last command reads the committed `results/demo_sweep.csv` and creates the
complete scoped figure manifest under `results/figures/`. The source CSV is an
explicitly synthetic demonstration dataset, not a hardware benchmark.

On a supported GPU, smoke-test either serving backend:

```bash
python scripts/smoke_test.py --config configs/model_qwen15moe.yaml
python scripts/smoke_test.py --backend sglang --config configs/model_qwen15moe.yaml
```

Run a configured serving + trace sweep with:

```bash
python -c "from moe_profiler.runner.sweep import run_sweep; print(run_sweep('configs/sweep_default.yaml'))"
```

That command loads the model once in trace mode, starts the configured serving
backend, and writes a revision-stamped CSV under `results/`. It can require model
access, substantial RAM/VRAM, and a backend-specific environment.

## Metrics

| Metric | Definition used here |
| --- | --- |
| TTFT | Wall time from request start to the first streamed token. |
| TPOT | `(end-to-end − TTFT) / (output tokens − 1)`; a one-token completion reports `0` with a warning because no inter-token interval exists. |
| Throughput | Output tokens divided by full batch wall time, including prefill; it is deliberately end-to-end throughput. |
| KV bytes/token | `2 × layers × KV heads × head dimension × dtype bytes`. |
| Activated ratio | Union of experts inside one forward pass, averaged across representative passes; passes are not cumulatively unioned. |
| MBU / S-MBU | Estimated bytes moved per TPOT divided by peak bandwidth, using total / activated model bytes respectively. |
| MFU / S-MFU | Token throughput × total / active FLOPs per token divided by peak FLOPs. |
| Arithmetic intensity | FLOPs divided by bytes moved; points left/right of the device ridge are memory/compute bound. |

All result rows include a model revision and backend version. The sweep seeds its
controlled prompt generation. Cached activation matrices live under
`results/activations/` and generated outputs remain ignored by Git.

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
- Expert-module discovery uses architecture heuristics and should be checked when
  adding a new model family.
- Profiler kernels do not always expose DRAM-byte counters. The code reports the
  counter source/fallback and should not turn an estimate into a measurement.
- Device peaks and prices are approximate reference values, not live quotes.
- The profiler does not yet cover quantisation accuracy, CAP radar plots, agentic
  session traffic, prefix-cache hit rate, or power measurement (Stages 9–12).
