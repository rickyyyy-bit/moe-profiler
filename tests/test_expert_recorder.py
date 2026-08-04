from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from moe_profiler.backends.base import ExpertTraceRecorder
from moe_profiler.profiling.expert_recorder import ExpertActivationRecorder


class FakeRouter(nn.Module):
    def __init__(self, n_experts: int, top_k: int, offset: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.offset = offset

    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        first = (token_ids.reshape(-1) + self.offset) % self.n_experts
        second = (first + 1) % self.n_experts
        return {"selected_experts": torch.stack((first, second), dim=-1)}


class FakeMoeLayer(nn.Module):
    def __init__(self, n_experts: int, top_k: int, offset: int) -> None:
        super().__init__()
        self.router = FakeRouter(n_experts, top_k, offset)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.router(token_ids)
        return token_ids


class FakeMoeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.layers = nn.ModuleList([FakeMoeLayer(4, 2, offset) for offset in range(2)])
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            _name_or_path="tests/fake-moe",
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = input_ids
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class FakeTokenizer:
    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding and truncation
        return {"input_ids": torch.tensor([[int(prompt)] for prompt in prompts])}


class LogitMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gate(hidden)


class LogitMoeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.moe = LogitMoeBlock()
        self.layers = nn.ModuleList([layer])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[0].moe(hidden)


def test_hooks_record_expected_layer_expert_matrix() -> None:
    model = FakeMoeModel()
    recorder = ExpertActivationRecorder()

    recorder.attach(model)
    model(input_ids=torch.tensor([[0]]))

    assert isinstance(recorder, ExpertTraceRecorder)
    matrix = recorder.activation_matrix()
    assert matrix.shape == (2, 4)
    np.testing.assert_array_equal(matrix[0], [1, 1, 0, 0])
    np.testing.assert_array_equal(matrix[1], [0, 1, 1, 0])
    assert recorder.activated_ratio(1) == pytest.approx(3 / 5)


def test_float_router_logits_are_converted_to_top_k_indices() -> None:
    model = LogitMoeModel()
    recorder = ExpertActivationRecorder(n_layers=1, n_experts=4, top_k=2)

    recorder.attach(model)
    model(torch.tensor([[1.0, 1.0]]))

    np.testing.assert_array_equal(recorder.activation_matrix(), [[1, 1, 0, 0]])


def test_ratio_grows_with_batch_and_cache_is_reused(tmp_path) -> None:
    model = FakeMoeModel()
    tokenizer = FakeTokenizer()
    recorder = ExpertActivationRecorder(cache_dir=tmp_path)

    batch_one = recorder.run(
        model,
        tokenizer,
        ["0"],
        max_batches=1,
        batch_size=1,
        model_id="tests/fake-moe",
        use_cache=False,
    )
    ratio_one = recorder.activated_ratio(1)
    batch_32 = recorder.run(
        model,
        tokenizer,
        [str(index % 4) for index in range(32)],
        max_batches=1,
        batch_size=32,
        model_id="tests/fake-moe",
        use_cache=False,
    )
    ratio_32 = recorder.activated_ratio(32)

    assert np.count_nonzero(batch_one) == 4
    assert np.count_nonzero(batch_32) == 8
    assert ratio_one <= ratio_32
    cache_path = recorder.cache_path("tests/fake-moe", 32)
    assert cache_path == tmp_path / "tests--fake-moe_32.npy"
    assert cache_path.exists()

    cached_recorder = ExpertActivationRecorder(cache_dir=tmp_path)
    cached = cached_recorder.run(
        model,
        tokenizer,
        [],
        max_batches=1,
        batch_size=32,
        model_id="tests/fake-moe",
    )
    np.testing.assert_array_equal(cached, batch_32)
    assert cached_recorder.batches_processed == 0


def test_run_stops_after_activation_union_stabilizes(tmp_path) -> None:
    recorder = ExpertActivationRecorder(
        stabilization_threshold=0.1,
        stabilization_patience=2,
        min_batches=2,
        cache_dir=tmp_path,
    )

    recorder.run(
        FakeMoeModel(),
        FakeTokenizer(),
        ["0"] * 10,
        max_batches=10,
        use_cache=False,
    )

    assert recorder.batches_processed == 3
