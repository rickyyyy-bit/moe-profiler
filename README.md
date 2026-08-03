# moe-profiler

`moe-profiler` is a sparsity-aware inference characterisation tool for open
Mixture-of-Experts models. The initial package provides a typed, installable
foundation for serving backends, metric calculations, sweeps, and visualisation.
The Stage 1 vLLM backend measures time to first token (TTFT), time per output token
(TPOT), end-to-end latency, and output throughput from a streamed
OpenAI-compatible response.

## Install

Use Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
```

Install vLLM and SGLang separately in their own environments because the engines
can have conflicting dependency constraints. Hugging Face credentials must be
supplied through the `HF_TOKEN` environment variable; never put tokens in config
files.

## Stage 1 smoke test

On a machine with a supported GPU and the `vllm` command installed:

```bash
python scripts/smoke_test.py --config configs/model_qwen15moe.yaml
```

The script owns the server process and stops it on normal exit, failure, or an
interrupt. On a machine without vLLM, it exits cleanly with a backend-unavailable
message.
