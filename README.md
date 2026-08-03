# moe-profiler

`moe-profiler` is a sparsity-aware inference characterisation tool for open
Mixture-of-Experts models. The initial package provides a typed, installable
foundation for serving backends, metric calculations, sweeps, and visualisation.

## Install

Use Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
```

Install vLLM and SGLang separately in their own environments because the engines
can have conflicting dependency constraints. Hugging Face credentials must be
supplied through the `HF_TOKEN` environment variable; never put tokens in config
files.
