# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M3 sampler compatibility QA for the PR21047 Gumbel integration."""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.sampler_strategy import RequestGroupKey
from tensorrt_llm._torch.pyexecutor.sampler.watermark import (
    WatermarkHandler,
    request_carries_watermark,
)
from tensorrt_llm.watermark import WatermarkConfig
from tensorrt_llm.watermark.sampling import gumbel_sample_reference


def _request(identity=1, slot=0, *, watermarking=None, dummy=False, tokens=None, **settings):
    values = dict(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        top_p_decay=None,
        top_p_min=None,
        top_p_reset_ids=None,
        beam_width=1,
        beam_width_array=None,
        length_penalty=None,
        beam_search_diversity_rate=None,
        early_stopping=None,
        seed=715,
    )
    values.update(settings)
    token_list = tokens if tokens is not None else [11, 12, 13, 14]
    return NS(
        py_request_id=identity,
        py_client_id=identity + 100,
        py_seq_slot=slot,
        py_watermarking=watermarking,
        py_watermark_neutral=None,
        py_sampling_strategy=None,
        py_is_draft=False,
        py_beam_width=values["beam_width"],
        is_dummy=dummy,
        is_context_init_state=False,
        is_last_context_chunk=False,
        py_return_log_probs=False,
        py_orig_prompt_len=4,
        sampling_config=NS(**values),
        get_beam_width_by_iter=lambda **kwargs: values["beam_width"],
        get_tokens=lambda beam: list(token_list),
    )


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({}, True),
        ({"temperature": 0.0}, False),
        ({"top_k": 1}, False),
        ({"top_p": 0.0}, False),
        ({"temperature": None, "top_k": None, "top_p": None, "min_p": None}, False),
        ({"temperature": None, "top_k": None, "top_p": 0.95, "min_p": None}, True),
        ({"min_p": 0.1}, True),
        ({"beam_width": 2}, False),
        ({"beam_width_array": [1, 2]}, False),
    ],
)
def test_eligibility_uses_the_m3_sampling_classification(settings, expected):
    assert request_carries_watermark(WatermarkConfig(key=7), _request(**settings)) is expected


def test_request_optout_forcing_dummy_and_unconfigured_semantics():
    on = WatermarkConfig(key=7)
    forced = WatermarkConfig(key=7, allow_request_opt_out=False)
    assert not request_carries_watermark(None, NS())
    assert not request_carries_watermark(on, _request(watermarking=False))
    assert request_carries_watermark(forced, _request(watermarking=False))
    assert not request_carries_watermark(forced, _request(dummy=True))
    assert not request_carries_watermark(forced, _request(temperature=0.0))


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({}, True),
        ({"top_p": 0.9}, False),
        ({"top_k": 4}, False),
        ({"min_p": 0.1}, False),
        ({"top_p_decay": 0.9}, False),
    ],
)
def test_fast_watermark_neutrality_has_no_new_sampling_resolution_dependency(settings, expected):
    assert TorchSampler._watermark_row_is_neutral(NS(), _request(**settings)) is expected


def test_same_owner_overlap_is_preserved_then_explicit_resume_restores_host_prefix(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.sampler.watermark.prefer_pinned", lambda: False
    )
    tokens = [10, 11, 12, 13]
    request = _request(tokens=tokens)
    handler = WatermarkHandler(
        WatermarkConfig(key=7, context_width=3), max_num_sequences=2, max_seq_len=64, device="cpu"
    )
    handler.observe([request])
    rows = handler.plan_reference(torch.tensor([0]), torch.tensor([True]))
    handler.commit_reference(rows, torch.tensor([3], dtype=torch.int32))
    tail, count = handler._tail.clone(), handler._count.clone()
    handler.observe([request])
    torch.testing.assert_close(handler._tail, tail)
    torch.testing.assert_close(handler._count, count)
    tokens.append(77)
    handler.invalidate([request])
    torch.testing.assert_close(handler._tail, tail)
    torch.testing.assert_close(handler._count, count)
    assert not handler.is_enabled(0)
    handler.observe([request])
    assert handler._tail[0].tolist() == [12, 13, 77]
    assert handler._count[0].item() == 1


def test_stale_or_dummy_invalidation_cannot_reset_a_reused_slot(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.sampler.watermark.prefer_pinned", lambda: False
    )
    handler = WatermarkHandler(
        WatermarkConfig(key=7), max_num_sequences=2, max_seq_len=64, device="cpu"
    )
    old, current = _request(), _request(identity=2, tokens=[41, 42, 43, 44])
    handler.observe([old])
    handler.observe([current])
    tail = handler._tail.clone()
    handler.invalidate([old, _request(identity=3, dummy=True), _request(identity=4, slot=None)])
    assert handler._slot_owner[0]() is current and handler.is_enabled(0)
    torch.testing.assert_close(handler._tail, tail)


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
def test_grouped_seam_keeps_native_filtered_probs_and_optout_fallback(marked, return_log_probs):
    requests = [
        _request(slot=2, watermarking=None if marked else False),
        _request(identity=2, slot=0, watermarking=False),
    ]
    probs = torch.tensor([[0.0, 0.2, 0.3, 0.5], [0.4, 0.3, 0.2, 0.1]])
    original = torch.tensor([1, 2], dtype=torch.int32)
    calls = []
    key = RequestGroupKey(strategy_key="temperature", needs_probs=return_log_probs)
    group = NS(
        indices=torch.tensor([0, 1]),
        strategies=[("temperature", 1.0)] * 2,
        speculation_needs_probs_indices=torch.tensor([], dtype=torch.int64),
        need_processed_logprobs=torch.tensor([return_log_probs, return_log_probs]),
        metadata=None,
    )

    class Native:
        @staticmethod
        def strategy_grouping_key(strategy):
            return strategy[0]

        @staticmethod
        def get_metadata_type_for_group(key):
            return None

        @staticmethod
        def sample_grouped_strategies(*args, **kwargs):
            calls.append(kwargs["return_probs"])
            return original.clone(), probs if kwargs["return_probs"] else None, None

    class Mark:
        def __init__(self):
            self.draws = 0

        def invalidate(self, requests):
            assert not requests

        def observe(self, requests, **kwargs):
            self.enabled = [r.py_watermarking is not False for r in requests]

        def step_inputs(self, slots):
            assert slots == [2, 0]
            return (
                (torch.tensor(slots), torch.tensor(self.enabled), False, True)
                if any(self.enabled)
                else None
            )

        def sample_step(self, values, temperature, slots, enabled, fallback, *, from_probs):
            assert values is probs and temperature is None and from_probs
            torch.testing.assert_close(fallback, original)
            self.draws += 1
            return torch.where(enabled, torch.full_like(fallback, 3), fallback)

    handler = Mark()
    grouped = {key: group}
    sampler = NS(
        _watermark=handler,
        _watermark_row_is_neutral=lambda r: True,
        _seed_manager=NS(
            observe=lambda r: None,
            take_row_seeds=lambda slots, **kw: NS(seed=torch.zeros(2), offset=torch.zeros(2)),
        ),
        _request_grouper=NS(
            group_requests_by_strategy_key=lambda *a, **kw: (grouped, torch.tensor([False, False]))
        ),
        _grouped_sampler_cls=Native,
        _add_metadata_to_grouped_requests=lambda *a, **kw: grouped,
        get_generator=lambda device: torch.Generator(device=device),
        max_beam_width=1,
        _use_beam_search=False,
    )
    before = probs.clone()
    result = TorchSampler._sample_batched_by_strategy(
        sampler,
        probs.log(),
        requests,
        {},
        logits_cuda_indexer=None,
        req_num_generated_tokens=torch.ones(2, dtype=torch.int32),
        req_num_steps=torch.ones(2, dtype=torch.int32),
        req_offsets=torch.arange(2),
        seq_slots=torch.tensor([2, 0]),
        seq_slots_cuda=torch.tensor([2, 0]),
        seq_lens_cuda=torch.ones(2),
        token_dtype=torch.int32,
        return_log_probs=return_log_probs,
    )
    assert calls == [marked or return_log_probs] and handler.draws == int(marked)
    assert result.next_tokens_cuda_int[:, 0].tolist() == ([3, 2] if marked else [1, 2])
    torch.testing.assert_close(probs, before, rtol=0, atol=0)
    if return_log_probs:
        assert result.processed_logprobs_end == 2
        torch.testing.assert_close(result.logprobs_cuda, probs.log())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires the new exact image on a native GPU"
)
def test_native_fast_graph_matches_pr_oracle_optout_and_padding():
    from tensorrt_llm._torch.pyexecutor.sampler.ops.custom import fused_sample_from_logits

    config = WatermarkConfig(key=7, deduplicate_contexts="none")
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=64,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=1,
            max_total_draft_tokens=0,
            watermark_config=config,
        )
    )
    requests = [
        _request(slot=2, temperature=0.7),
        _request(identity=2, slot=0, watermarking=False),
        _request(identity=3, slot=None, dummy=True),
    ]
    slots = [2, 0, 4]
    destinations = torch.tensor(slots, device="cuda")
    logits = torch.tensor([[0.0, -float("inf"), 0.4, 1.0]] * 3, device="cuda")
    sampler.stage_fast_sampling_params(requests, slots, [1, 1, 1], 2)
    assert sampler._fast_watermark_enabled[:3].tolist() == [True, False, False]
    for _ in range(3):
        sampler._fast_sample_in_graph(logits, destinations, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampler._fast_sample_in_graph(logits, destinations, None)
    sampler._watermark.invalidate(requests[:2])
    sampler._watermark.observe(requests[:2], is_neutral=sampler._watermark_row_is_neutral)
    for step in range(3):
        sampler.stage_fast_sampling_params(requests, slots, [1, 1, 1], 2)
        rows = sampler._watermark.plan_reference(destinations, sampler._fast_watermark_enabled[:3])
        probabilities = (logits / sampler._fast_temperatures[:3, None]).softmax(-1)
        keyed = gumbel_sample_reference(probabilities, rows.seeds[:, 0])
        native = fused_sample_from_logits(
            logits,
            sampler._fast_temperatures[:3],
            sampler._fast_top_ks[:3],
            sampler._fast_top_ps[:3],
            sampler._fast_min_ps[:3],
            seed=sampler._fast_seeds[:3],
            offset=sampler._fast_offsets[:3],
        )
        expected = torch.where(rows.active[:, 0], keyed, native)
        graph.replay()
        torch.testing.assert_close(
            sampler.store.new_tokens[0, destinations, 0], expected, rtol=0, atol=0
        )
        assert sampler._watermark._count[[2, 0, 4]].tolist() == [step + 1, 0, 0]
