"""Start one serving backend and print streamed latency metrics."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import yaml
from pydantic import ValidationError

from moe_profiler.backends import get_backend
from moe_profiler.backends.base import BackendError, BackendUnavailableError
from moe_profiler.config import ModelConfig
from moe_profiler.workloads.base import RequestSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a model YAML configuration file.",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "sglang"),
        default="vllm",
        help="Serving engine to launch (default: vllm).",
    )
    return parser.parse_args()


def load_config(path: Path) -> ModelConfig:
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read config {path}: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise ValueError(f"config {path} must contain a YAML mapping")
    try:
        return ModelConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"invalid config {path}: {exc}") from exc


def run(config: ModelConfig, backend_name: str) -> None:
    backend = get_backend(
        backend_name,
        host=config.host,
        port=config.port,
        startup_timeout_s=config.startup_timeout_s,
    )
    try:
        backend.start(
            config.model_id,
            quant=config.quant,
            tp_size=config.tp_size,
            dtype=config.dtype,
        )
        request = RequestSpec(
            request_id=str(uuid.uuid4()),
            prompt=(
                "Explain in three concise sentences why sparse Mixture-of-Experts "
                "models can reduce computation per generated token."
            ),
            max_tokens=96,
        )
        result = backend.generate([request])[0]
        throughput_tok_s = result.output_tokens / result.e2e_s
        print(f"ttft_s={result.ttft_s:.6f}")
        print(f"tpot_s={result.tpot_s:.6f}")
        print(f"throughput_tok_s={throughput_tok_s:.6f}")
    finally:
        backend.stop()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        run(config, args.backend)
    except BackendUnavailableError as exc:
        print(f"backend unavailable: {exc}", file=sys.stderr)
        return 2
    except (BackendError, ValueError) as exc:
        print(f"smoke test failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("smoke test interrupted; server stopped", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
