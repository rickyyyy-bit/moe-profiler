"""Record routed expert activations from Hugging Face MoE forward passes."""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from moe_profiler.provenance import ExecutionPhase, TraceCacheIdentity

_LAYER_INDEX = re.compile(r"(?:^|\.)(?:layers?|blocks?|h)\.(\d+)(?:\.|$)")
_EXPERT_KEYS = (
    "selected_experts",
    "expert_indices",
    "router_indices",
    "topk_indices",
)
_CONFIG_EXPERT_FIELDS = (
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "n_experts",
)
_CONFIG_TOP_K_FIELDS = (
    "num_experts_per_tok",
    "num_selected_experts",
    "moe_top_k",
    "top_k",
)


@dataclass(frozen=True)
class _RouterBinding:
    layer_index: int
    name: str
    module: nn.Module


class ActivationRecord(BaseModel):
    """Routing observation for one layer in one instrumented execution step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: ExecutionPhase
    step_index: int = Field(ge=0)
    layer_index: int = Field(ge=0)
    routed_expert_assignment_counts: tuple[int, ...]
    distinct_routed_experts: int = Field(ge=0)
    distinct_routed_expert_fraction: float = Field(ge=0.0, le=1.0)
    total_token_assignments: int = Field(ge=0)
    shared_expert_assignment_counts: tuple[int, ...]
    max_to_mean_load_ratio: float = Field(ge=0.0)
    coefficient_of_variation: float = Field(ge=0.0)
    normalized_routing_entropy: float = Field(ge=0.0, le=1.0)


class ExpertActivationRecorder:
    """Collect per-layer routed-expert counts using router forward hooks.

    Router modules differ across model families. The recorder recognizes named
    router/gate modules and router-like class names, then accepts either explicit
    integer expert indices or floating-point router logits as hook output.
    """

    def __init__(
        self,
        *,
        n_layers: int | None = None,
        n_experts: int | None = None,
        top_k: int | None = None,
        n_shared_experts: int | None = None,
        stabilization_threshold: float = 0.01,
        stabilization_patience: int = 2,
        min_batches: int = 2,
        cache_dir: str | Path = "results/activations",
    ) -> None:
        if n_layers is not None and n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if n_experts is not None and n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= stabilization_threshold <= 1.0:
            raise ValueError("stabilization_threshold must be between 0 and 1")
        if stabilization_patience <= 0:
            raise ValueError("stabilization_patience must be positive")
        if min_batches <= 0:
            raise ValueError("min_batches must be positive")

        self._requested_n_layers = n_layers
        self._requested_n_experts = n_experts
        self._requested_top_k = top_k
        self._requested_shared_experts = n_shared_experts
        self.stabilization_threshold = stabilization_threshold
        self.stabilization_patience = stabilization_patience
        self.min_batches = min_batches
        self.cache_dir = Path(cache_dir)

        self.n_layers: int | None = None
        self.n_experts: int | None = None
        self.top_k: int | None = None
        self.n_shared_experts = 0
        self.batches_processed = 0

        self._model: nn.Module | None = None
        self._bindings: list[_RouterBinding] = []
        self._router_layer_indices: tuple[int, ...] = ()
        self._handles: list[Any] = []
        self._counts: NDArray[np.int64] | None = None
        self._counts_by_batch: dict[int, NDArray[np.int64]] = {}
        self._ratios_by_batch: dict[int, float] = {}
        self._current_batch_size = 1
        self._records: list[ActivationRecord] = []

    def attach(self, model: object) -> None:
        """Locate MoE routers in ``model`` and attach activation hooks."""
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        self.detach()
        candidates = [
            (name, module)
            for name, module in model.named_modules()
            if name and _is_router_candidate(name, module)
        ]
        if not candidates:
            raise ValueError("no MoE router/gate modules were found in the model")

        bindings = _bind_router_layers(candidates)
        config = getattr(model, "config", None)
        inferred_layers = _config_int(config, "num_hidden_layers", "n_layer")
        inferred_experts = _config_int(config, *_CONFIG_EXPERT_FIELDS)
        inferred_top_k = _config_int(config, *_CONFIG_TOP_K_FIELDS)
        inferred_shared = _config_int(
            config, "n_shared_experts", "num_shared_experts", allow_zero=True
        )

        n_layers = self._requested_n_layers or inferred_layers
        if n_layers is None:
            n_layers = max(binding.layer_index for binding in bindings) + 1
        n_experts = self._requested_n_experts or inferred_experts
        if n_experts is None:
            n_experts = _module_expert_count(bindings)
        if n_experts is None:
            raise ValueError(
                "could not infer n_experts; pass it to ExpertActivationRecorder"
            )
        top_k = self._requested_top_k or inferred_top_k
        if top_k is None:
            top_k = _module_top_k(bindings) or 1
        shared_experts = self._requested_shared_experts
        if shared_experts is None:
            shared_experts = inferred_shared or 0

        if max(binding.layer_index for binding in bindings) >= n_layers:
            raise ValueError("router layer index falls outside n_layers")
        if top_k > n_experts:
            raise ValueError("top_k cannot exceed n_experts")

        self._model = model
        self._bindings = bindings
        self._router_layer_indices = tuple(
            sorted({binding.layer_index for binding in bindings})
        )
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_shared_experts = shared_experts
        self.reset(batch_size=1, clear_history=True)

        for binding in bindings:
            handle = binding.module.register_forward_hook(
                self._make_hook(binding.layer_index)
            )
            self._handles.append(handle)

    def detach(self) -> None:
        """Remove all installed hooks while retaining recorded counts."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bindings.clear()
        self._model = None

    def reset(self, *, batch_size: int = 1, clear_history: bool = False) -> None:
        """Clear current counts and select the batch-size history bucket."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_layers is None or self.n_experts is None:
            raise RuntimeError("attach a model before resetting the recorder")
        self._current_batch_size = batch_size
        self._counts = np.zeros((self.n_layers, self.n_experts), dtype=np.int64)
        self._records.clear()
        if clear_history:
            self._counts_by_batch.clear()
            self._ratios_by_batch.clear()
        else:
            self._ratios_by_batch.pop(batch_size, None)
        self._counts_by_batch[batch_size] = self._counts.copy()
        self.batches_processed = 0

    def run(
        self,
        model: object,
        tokenizer: object,
        dataset: Iterable[object],
        max_batches: int,
        *,
        batch_size: int = 1,
        model_id: str | None = None,
        use_cache: bool = True,
        cache_identity: TraceCacheIdentity | None = None,
        phase: ExecutionPhase = "prefill",
    ) -> NDArray[np.int64]:
        """Trace dataset batches until the mean per-pass ratio stabilizes.

        Each forward pass measures the expert union within that pass only. The
        reported ratio is the mean of those pass ratios; cumulative counts are
        retained separately for the activation matrix. Tracing stops when the
        running mean changes by less than ``stabilization_threshold`` for
        ``stabilization_patience`` consecutive batches.
        """
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if self._model is not model or not self._handles:
            self.attach(model)

        del model_id  # Cache reuse requires the complete typed identity below.
        if use_cache and cache_identity is not None:
            if cache_identity.concurrency != batch_size:
                raise ValueError("cache identity concurrency does not match batch_size")
            if cache_identity.top_k != self._require_top_k():
                raise ValueError("cache identity top_k does not match recorder")
            cached = self.load_cache(cache_identity)
            if cached is not None:
                return cached

        self.reset(batch_size=batch_size)
        pass_ratios: list[float] = []
        previous_average: float | None = None
        stable_batches = 0
        batches = _batched_prompts(dataset, batch_size)
        model_was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for prompts in batches:
                    if self.batches_processed >= max_batches:
                        break
                    inputs = _tokenize(tokenizer, prompts)
                    inputs = _move_to_model_device(inputs, model)
                    counts_before_pass = self.activation_matrix()
                    model(**inputs)
                    self.batches_processed += 1

                    pass_counts = self.activation_matrix() - counts_before_pass
                    self._records.extend(
                        _activation_records(
                            pass_counts,
                            phase=phase,
                            step_index=self.batches_processed - 1,
                            top_k=self._require_top_k(),
                            n_shared_experts=self.n_shared_experts,
                            layer_indices=self._router_layer_indices,
                        )
                    )
                    pass_ratios.append(self._ratio_from_counts(pass_counts))
                    running_average = statistics.fmean(pass_ratios)
                    average_change = (
                        float("inf")
                        if previous_average is None
                        else abs(running_average - previous_average)
                    )
                    if (
                        self.batches_processed >= self.min_batches
                        and average_change < self.stabilization_threshold
                    ):
                        stable_batches += 1
                    else:
                        stable_batches = 0
                    previous_average = running_average
                    if stable_batches >= self.stabilization_patience:
                        break
        finally:
            model.train(model_was_training)

        if self.batches_processed == 0:
            raise ValueError("dataset did not yield any prompts")
        self._counts_by_batch[batch_size] = self.activation_matrix()
        self._ratios_by_batch[batch_size] = statistics.fmean(pass_ratios)
        if cache_identity is not None:
            self.save_cache(cache_identity)
        return self.activation_matrix()

    def activation_matrix(self) -> NDArray[np.int64]:
        """Return a copy of the ``[n_layers, n_experts]`` count matrix."""
        if self._counts is None:
            raise RuntimeError("attach a model before reading activations")
        return self._counts.copy()

    def activated_ratio(self, batch_size: int) -> float:
        """Return activated experts divided by routed plus shared experts."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ratio = self._ratios_by_batch.get(batch_size)
        if ratio is not None:
            return ratio
        try:
            counts = self._counts_by_batch[batch_size]
        except KeyError as exc:
            raise ValueError(
                f"no activations recorded for batch size {batch_size}"
            ) from exc
        return self._ratio_from_counts(counts)

    def activation_records(self) -> tuple[ActivationRecord, ...]:
        """Return immutable per-step records; the matrix remains a derived summary."""
        return tuple(self._records)

    def cache_path(self, identity: TraceCacheIdentity) -> Path:
        """Return the content-addressed NPZ path for a complete trace identity."""
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "--", identity.model_id.strip())
        return self.cache_dir / f"{safe_model}_{identity.digest()}.npz"

    def save_cache(self, identity: TraceCacheIdentity) -> Path:
        """Persist counts, per-step observations, and exact identity together."""
        batch_size = identity.concurrency
        try:
            counts = self._counts_by_batch[batch_size]
        except KeyError as exc:
            raise ValueError(
                f"no activations recorded for batch size {batch_size}"
            ) from exc
        path = self.cache_path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        ratio = self._ratios_by_batch.get(batch_size)
        if ratio is None:
            ratio = self._ratio_from_counts(counts)
            self._ratios_by_batch[batch_size] = ratio
        records_json = "\n".join(record.model_dump_json() for record in self._records)
        np.savez_compressed(
            path,
            counts=counts,
            ratio=np.asarray(ratio, dtype=np.float64),
            metadata=np.asarray(identity.model_dump_json()),
            records=np.asarray(records_json),
        )
        return path

    def load_cache(self, identity: TraceCacheIdentity) -> NDArray[np.int64] | None:
        """Load an exactly compatible cached trace, or return ``None``."""
        batch_size = identity.concurrency
        path = self.cache_path(identity)
        if not path.exists():
            return None
        expected_shape = (self._require_n_layers(), self._require_n_experts())
        with np.load(path, allow_pickle=False) as archive:
            required = {"counts", "ratio", "metadata", "records"}
            if not required <= set(archive.files):
                raise ValueError("cached activation trace is missing required fields")
            counts = archive["counts"]
            cached_ratio = archive["ratio"]
            cached_identity = TraceCacheIdentity.model_validate_json(
                str(archive["metadata"].item())
            )
            records_text = str(archive["records"].item())
        if cached_identity != identity:
            raise ValueError("cached activation provenance is incompatible")
        if counts.shape != expected_shape:
            message = (
                f"cached activation shape {counts.shape} does not match "
                f"{expected_shape}"
            )
            raise ValueError(message)
        if not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("cached activation sheet must contain integer counts")
        if cached_ratio.shape != () or not np.issubdtype(
            cached_ratio.dtype, np.floating
        ):
            raise ValueError("cached activation ratio must be a floating-point scalar")
        ratio = float(cached_ratio.item())
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("cached activation ratio must be between 0 and 1")
        self._current_batch_size = batch_size
        self._counts = counts.astype(np.int64, copy=True)
        self._counts_by_batch[batch_size] = self._counts.copy()
        self._ratios_by_batch[batch_size] = ratio
        self._records = [
            ActivationRecord.model_validate_json(line)
            for line in records_text.splitlines()
            if line
        ]
        self.batches_processed = 0
        return self.activation_matrix()

    def _ratio_from_counts(self, counts: NDArray[np.int64]) -> float:
        rows = self._router_layer_indices
        if not rows:
            raise RuntimeError("attach a model before computing activation ratios")
        routed_active = int(np.count_nonzero(counts[list(rows)]))
        shared_active = len(rows) * self.n_shared_experts
        experts_per_layer = self._require_n_experts() + self.n_shared_experts
        return (routed_active + shared_active) / (len(rows) * experts_per_layer)

    def _make_hook(self, layer_index: int) -> Any:
        def record(
            _module: nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            indices = _expert_indices(
                output,
                n_experts=self._require_n_experts(),
                top_k=self._require_top_k(),
            )
            flattened = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            valid = flattened[
                (flattened >= 0) & (flattened < self._require_n_experts())
            ]
            if valid.numel() == 0:
                return
            additions = torch.bincount(
                valid, minlength=self._require_n_experts()
            ).numpy()
            if self._counts is None:
                raise RuntimeError("recorder counts were not initialized")
            self._counts[layer_index] += additions
            self._counts_by_batch[self._current_batch_size] = self._counts.copy()

        return record

    def _require_n_layers(self) -> int:
        if self.n_layers is None:
            raise RuntimeError("attach a model before tracing")
        return self.n_layers

    def _require_n_experts(self) -> int:
        if self.n_experts is None:
            raise RuntimeError("attach a model before tracing")
        return self.n_experts

    def _require_top_k(self) -> int:
        if self.top_k is None:
            raise RuntimeError("attach a model before tracing")
        return self.top_k


def _is_router_candidate(name: str, module: nn.Module) -> bool:
    lowered_name = name.lower()
    class_name = type(module).__name__.lower()
    leaf = lowered_name.rsplit(".", maxsplit=1)[-1]
    if "shared_expert" in lowered_name or "gate_proj" in lowered_name:
        return False
    if "router" in class_name or ("topk" in class_name and "gate" in class_name):
        return True
    if leaf == "router":
        return True
    context_is_moe = any(
        marker in lowered_name
        for marker in ("moe", "expert", "sparse", "mlp", "feed_forward")
    )
    return leaf == "gate" and context_is_moe


def _activation_records(
    counts: NDArray[np.int64],
    *,
    phase: ExecutionPhase,
    step_index: int,
    top_k: int,
    n_shared_experts: int,
    layer_indices: Sequence[int],
) -> list[ActivationRecord]:
    """Convert one forward pass into per-layer routing statistics.

    Entropy uses ``-sum(p * log(p)) / log(n_experts)``. Empty layers and
    single-expert layers report zero because no routing diversity was observed.
    """
    records: list[ActivationRecord] = []
    n_experts = counts.shape[1]
    for layer_index in layer_indices:
        row = counts[layer_index].astype(np.float64, copy=False)
        total_assignments = int(row.sum())
        distinct = int(np.count_nonzero(row))
        mean = float(row.mean()) if row.size else 0.0
        max_to_mean = float(row.max() / mean) if mean > 0 else 0.0
        coefficient = float(row.std() / mean) if mean > 0 else 0.0
        if total_assignments > 0 and n_experts > 1:
            probabilities = row[row > 0] / total_assignments
            entropy = float(
                -np.sum(probabilities * np.log(probabilities)) / np.log(n_experts)
            )
        else:
            entropy = 0.0
        routed_tokens = total_assignments // top_k
        records.append(
            ActivationRecord(
                phase=phase,
                step_index=step_index,
                layer_index=layer_index,
                routed_expert_assignment_counts=tuple(int(value) for value in row),
                distinct_routed_experts=distinct,
                distinct_routed_expert_fraction=distinct / n_experts,
                total_token_assignments=total_assignments,
                shared_expert_assignment_counts=(routed_tokens,) * n_shared_experts,
                max_to_mean_load_ratio=max_to_mean,
                coefficient_of_variation=coefficient,
                normalized_routing_entropy=entropy,
            )
        )
    return records


def _bind_router_layers(
    candidates: Sequence[tuple[str, nn.Module]],
) -> list[_RouterBinding]:
    selected: dict[int, tuple[int, str, nn.Module]] = {}
    unindexed: list[tuple[str, nn.Module]] = []
    for name, module in candidates:
        match = _LAYER_INDEX.search(name)
        if match is None:
            unindexed.append((name, module))
            continue
        layer_index = int(match.group(1))
        priority = _router_priority(name, module)
        previous = selected.get(layer_index)
        if previous is None or priority > previous[0]:
            selected[layer_index] = (priority, name, module)

    next_index = 0
    for name, module in unindexed:
        while next_index in selected:
            next_index += 1
        selected[next_index] = (_router_priority(name, module), name, module)
        next_index += 1

    return [
        _RouterBinding(layer_index=index, name=value[1], module=value[2])
        for index, value in sorted(selected.items())
    ]


def _router_priority(name: str, module: nn.Module) -> int:
    class_name = type(module).__name__.lower()
    if "router" in class_name:
        return 3
    if name.lower().endswith(".router"):
        return 2
    return 1


def _expert_indices(output: object, *, n_experts: int, top_k: int) -> torch.Tensor:
    integer = _find_integer_tensor(output)
    if integer is not None:
        return integer
    logits = _find_router_logits(output, n_experts)
    if logits is None:
        raise ValueError(
            "router hook output contained neither expert indices nor router logits"
        )
    return torch.topk(logits, k=top_k, dim=-1).indices


def _find_integer_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, Mapping):
        for key in _EXPERT_KEYS:
            candidate = value.get(key)
            if (
                isinstance(candidate, torch.Tensor)
                and not candidate.dtype.is_floating_point
            ):
                return candidate
        for candidate in value.values():
            found = _find_integer_tensor(candidate)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for candidate in reversed(value):
            found = _find_integer_tensor(candidate)
            if found is not None:
                return found
    elif isinstance(value, torch.Tensor) and not value.dtype.is_floating_point:
        return value
    return None


def _find_router_logits(value: object, n_experts: int) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        if (
            value.dtype.is_floating_point
            and value.ndim > 0
            and value.shape[-1] == n_experts
        ):
            return value
        return None
    candidates: Iterable[object]
    if isinstance(value, Mapping):
        candidates = value.values()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        candidates = reversed(value)
    else:
        return None
    for candidate in candidates:
        found = _find_router_logits(candidate, n_experts)
        if found is not None:
            return found
    return None


def _module_expert_count(bindings: Sequence[_RouterBinding]) -> int | None:
    values = {
        value
        for binding in bindings
        if (value := _positive_int_attr(binding.module, *_CONFIG_EXPERT_FIELDS))
        is not None
    }
    for binding in bindings:
        out_features = getattr(binding.module, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            values.add(out_features)
    if len(values) > 1:
        raise ValueError(f"routers disagree on n_experts: {sorted(values)}")
    return next(iter(values), None)


def _module_top_k(bindings: Sequence[_RouterBinding]) -> int | None:
    values = {
        value
        for binding in bindings
        if (value := _positive_int_attr(binding.module, *_CONFIG_TOP_K_FIELDS))
        is not None
    }
    if len(values) > 1:
        raise ValueError(f"routers disagree on top_k: {sorted(values)}")
    return next(iter(values), None)


def _positive_int_attr(value: object, *names: str) -> int | None:
    for name in names:
        candidate = getattr(value, name, None)
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate > 0
        ):
            return candidate
    return None


def _config_int(config: object, *names: str, allow_zero: bool = False) -> int | None:
    if config is None:
        return None
    for name in names:
        value = getattr(config, name, None)
        minimum = 0 if allow_zero else 1
        if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
            return value
    return None


def _batched_prompts(dataset: Iterable[object], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for sample in dataset:
        batch.append(_prompt_from_sample(sample))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _prompt_from_sample(sample: object) -> str:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, Mapping):
        for key in ("prompt", "question", "text"):
            value = sample.get(key)
            if isinstance(value, str):
                return value
    raise TypeError("dataset samples must be strings or contain prompt/question/text")


def _tokenize(tokenizer: object, prompts: list[str]) -> dict[str, torch.Tensor]:
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer must return a mapping of model inputs")
    inputs = dict(encoded)
    if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
        raise TypeError("all tokenizer model inputs must be torch tensors")
    return inputs


def _move_to_model_device(
    inputs: dict[str, torch.Tensor], model: nn.Module
) -> dict[str, torch.Tensor]:
    parameter = next(iter(model.parameters()), None)
    if parameter is None:
        return inputs
    return {name: value.to(parameter.device) for name, value in inputs.items()}
