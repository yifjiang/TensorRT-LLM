# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend routing and native input contracts without allocating CUDA storage."""

from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch.speculative import interface

SamplingInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _inputs(batch: int = 2, drafts: int = 3, vocab: int = 16) -> SamplingInputs:
    device = torch.device("cuda", 0)
    return (
        torch.empty((batch, drafts, vocab), dtype=torch.float32, device=device),
        torch.empty((batch, drafts), dtype=torch.int32, device=device),
        torch.empty((batch, drafts + 1, vocab), dtype=torch.float32, device=device),
        torch.empty(batch, dtype=torch.int64, device=device),
        torch.empty(batch, dtype=torch.int64, device=device),
    )


@dataclass
class _Route:
    inputs: SamplingInputs
    native: Mock
    stock: Mock


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Route]:
    with FakeTensorMode():
        arguments = _inputs()
        tokens = torch.empty((2, 4), dtype=torch.int32, device="cuda:0")
        counts = torch.empty(2, dtype=torch.int32, device="cuda:0")
        native = Mock(return_value=(tokens, counts))
        stock = Mock(return_value=(tokens, counts, counts))
        monkeypatch.setattr(interface, "IS_FLASHINFER_AVAILABLE", True)
        monkeypatch.setattr(
            interface,
            "flashinfer",
            SimpleNamespace(
                __version__="0.6.18", sampling=SimpleNamespace(chain_speculative_sampling=stock)
            ),
        )
        monkeypatch.setattr(interface, "_CHAIN_SPEC_OVERRIDE", None)
        monkeypatch.setattr(torch.ops, "trtllm", SimpleNamespace(chain_speculative_sampling=native))
        yield _Route(arguments, native, stock)


def _sample(
    arguments: SamplingInputs, *, deterministic: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    draft_probs, tokens, target_probs, seed, offset = arguments
    return interface.rejection_sampling_one_model(
        draft_probs, tokens, target_probs, deterministic=deterministic, seed=seed, offset=offset
    )


@pytest.mark.parametrize("batch,drafts", [(2, 3), (1, 3), (2, 0)])
def test_per_request_state_keeps_native_arguments_and_counts(
    route: _Route, batch: int, drafts: int
) -> None:
    arguments = _inputs(batch=batch, drafts=drafts)
    assert _sample(arguments) is route.native.return_value
    route.native.assert_called_once()
    assert all(
        actual is expected for actual, expected in zip(route.native.call_args.args, arguments)
    )
    route.stock.assert_not_called()


def test_model_override_keeps_priority(route: _Route, monkeypatch: pytest.MonkeyPatch) -> None:
    override = Mock(return_value=route.native.return_value)
    monkeypatch.setattr(interface, "_CHAIN_SPEC_OVERRIDE", override)
    arguments = list(route.inputs)
    arguments[3] = arguments[3].to("cpu")
    assert _sample(tuple(arguments)) is override.return_value
    override.assert_called_once()
    route.native.assert_not_called()
    route.stock.assert_not_called()


def test_declined_model_override_uses_native_route(
    route: _Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = Mock(return_value=None)
    monkeypatch.setattr(interface, "_CHAIN_SPEC_OVERRIDE", override)
    assert _sample(route.inputs) is route.native.return_value
    override.assert_called_once()
    route.native.assert_called_once()
    route.stock.assert_not_called()


@pytest.mark.parametrize("unsigned_fields", [(3,), (4,), (3, 4)])
def test_unsigned_request_state_reaches_native_as_same_storage_view(
    route: _Route, unsigned_fields: tuple[int, ...]
) -> None:
    arguments = list(route.inputs)
    for index in unsigned_fields:
        arguments[index] = arguments[index].to(torch.uint64)
    assert _sample(tuple(arguments)) is route.native.return_value
    for index in (3, 4):
        actual = route.native.call_args.args[index]
        assert actual.dtype == torch.int64
        assert actual.untyped_storage()._cdata == arguments[index].untyped_storage()._cdata
    route.stock.assert_not_called()


def test_unsigned_state_high_bits_are_reinterpreted_without_copy() -> None:
    state = torch.tensor([0, 2**63 - 1, 2**63, 2**64 - 1], dtype=torch.uint64)
    native = interface._native_rng_state(state)
    assert native.data_ptr() == state.data_ptr()
    torch.testing.assert_close(
        native, torch.tensor([0, 2**63 - 1, -(2**63), -1], dtype=torch.int64)
    )


def test_signed_state_keeps_tensor_identity() -> None:
    state = torch.tensor([0, -1], dtype=torch.int64)
    assert interface._native_rng_state(state) is state


@pytest.mark.parametrize("kind", ["default", "scalar", "broadcast", "nondeterministic"])
def test_stock_sampling_modes_keep_existing_backend(route: _Route, kind: str) -> None:
    draft_probs, tokens, target_probs, seed, offset = route.inputs
    if kind == "default":
        seed = offset = None
    elif kind == "scalar":
        seed, offset = 123, 512
    elif kind == "broadcast":
        seed = torch.empty(1, dtype=torch.int64, device="cuda:0")
        offset = torch.empty(1, dtype=torch.int64, device="cuda:0")
    interface.rejection_sampling_one_model(
        draft_probs,
        tokens,
        target_probs,
        deterministic=kind != "nondeterministic",
        seed=seed,
        offset=offset,
    )
    route.stock.assert_called_once()
    assert route.stock.call_args.kwargs["seed"] is seed
    assert route.stock.call_args.kwargs["offset"] is offset
    route.native.assert_not_called()


@pytest.mark.parametrize(
    "case,error,message",
    [
        ("seed_dtype", TypeError, "seed.*torch.int64"),
        ("probability_dtype", TypeError, "draft_probs.*torch.float32"),
        ("token_dtype", TypeError, "draft_token_ids.*torch.int32"),
        ("seed_rank", ValueError, "one seed and offset"),
        ("offset_length", ValueError, "one seed and offset"),
        ("target_shape", ValueError, "shapes do not agree"),
        ("probability_rank", ValueError, "probabilities.*batch"),
        ("empty_batch", ValueError, "nonempty batch"),
        ("empty_vocab", ValueError, "nonempty batch"),
        ("cpu_seed", ValueError, "one CUDA device: seed"),
        ("other_device", ValueError, "one CUDA device: offset"),
        ("strided_probs", ValueError, "contiguous: draft_probs"),
        ("strided_seed", ValueError, "contiguous: seed"),
    ],
)
def test_invalid_native_contract_fails_without_stock_fallback(
    route: _Route, case: str, error: type[Exception], message: str
) -> None:
    arguments = list(route.inputs)
    if case == "seed_dtype":
        arguments[3] = arguments[3].to(torch.int32)
    elif case == "probability_dtype":
        arguments[0] = arguments[0].to(torch.float16)
    elif case == "token_dtype":
        arguments[1] = arguments[1].to(torch.int64)
    elif case == "seed_rank":
        arguments[3] = arguments[3].reshape(2, 1)
    elif case == "offset_length":
        arguments[4] = torch.empty(1, dtype=torch.int64, device="cuda:0")
    elif case == "target_shape":
        arguments[2] = torch.empty((2, 3, 16), device="cuda:0")
    elif case == "probability_rank":
        arguments[0] = arguments[0].flatten(0, 1)
    elif case == "empty_batch":
        arguments = list(_inputs(batch=0))
    elif case == "empty_vocab":
        arguments = list(_inputs(vocab=0))
    elif case == "cpu_seed":
        arguments[3] = arguments[3].to("cpu")
    elif case == "other_device":
        arguments[4] = arguments[4].to("cuda:1")
    elif case == "strided_probs":
        arguments[0] = torch.empty_strided((2, 3, 16), (96, 32, 2), device="cuda:0")
    elif case == "strided_seed":
        arguments[3] = torch.empty_strided((2,), (2,), dtype=torch.int64, device="cuda:0")
    with pytest.raises(error, match=message):
        _sample(tuple(arguments))
    route.native.assert_not_called()
    route.stock.assert_not_called()


def test_missing_native_operator_fails_without_stock_fallback(
    route: _Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.ops, "trtllm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="compiled trtllm::chain_speculative_sampling CUDA"):
        _sample(route.inputs)
    route.stock.assert_not_called()


def test_missing_native_cuda_implementation_fails_without_stock_fallback(route: _Route) -> None:
    route.native.side_effect = NotImplementedError("CUDA kernel unavailable")
    with pytest.raises(
        RuntimeError, match="CUDA implementation of trtllm::chain_speculative_sampling"
    ):
        _sample(route.inputs)
    route.stock.assert_not_called()


def test_native_kernel_error_is_not_hidden_by_stock_fallback(route: _Route) -> None:
    route.native.side_effect = RuntimeError("native kernel launch failed")
    with pytest.raises(RuntimeError, match="native kernel launch failed"):
        _sample(route.inputs)
    route.stock.assert_not_called()
