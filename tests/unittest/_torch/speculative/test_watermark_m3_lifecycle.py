# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-prefix and metadata ownership contracts of the M3 watermark adapter."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative import spec_sampler_base
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateSpec, SpecSampler
from tensorrt_llm._torch.speculative.utils import _watermark_unsupported_reason


class Handler:
    def __init__(self):
        self.invalidated = []
        self.observed = []

    def invalidate(self, requests):
        self.invalidated.extend(requests)

    def observe(self, requests):
        self.observed.extend(requests)

    def is_enabled(self, slot):
        return slot in {0, 1}


def test_graph_metadata_preserves_parent_watermark_storage():
    state = SimpleNamespace(ready=True, handler=Handler(),
                            enabled=torch.zeros(70, dtype=torch.bool))
    parent = SpecMetadata(max_num_requests=70, max_draft_len=7,
                          max_total_draft_tokens=7, watermark=state)
    graph = parent.create_cuda_graph_metadata(4)
    assert parent.max_num_requests == 70 and graph.max_num_requests == 4
    assert graph.watermark is parent.watermark
    assert graph.watermark.enabled.data_ptr() == state.enabled.data_ptr()


@pytest.mark.parametrize("algorithm,target_only,rejection,supported", [
    ("gumbel", True, True, True),
    ("gumbel", False, True, False),
    ("gumbel_dual_key", True, True, False),
    ("gumbel", True, False, False),
])
def test_dspark_admission_requires_explicit_target_only_rejection(
        monkeypatch, algorithm, target_only, rejection, supported):
    monkeypatch.delenv("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", raising=False)
    config = SimpleNamespace(spec_dec_mode=SimpleNamespace(is_dspark=lambda: True),
                              use_rejection_sampling=rejection)
    watermark = SimpleNamespace(algorithm=algorithm,
                                 allow_target_only_watermarking=target_only)
    reason = _watermark_unsupported_reason(config, watermark)
    assert (reason is None) == supported


def test_forced_acceptance_cannot_be_admitted_with_watermark(monkeypatch):
    monkeypatch.setenv("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", "2.5")
    reason = _watermark_unsupported_reason(SimpleNamespace(), SimpleNamespace())
    assert "forced speculative acceptance" in reason


def test_frozen_semantic_requests_reset_only_context_resumption(monkeypatch):
    handler = Handler()
    state = SimpleNamespace(handler=handler, ready=True,
                            enabled=torch.zeros(4, dtype=torch.bool))
    metadata = SimpleNamespace(watermark=state, watermark_ready=True)
    generation = SimpleNamespace(py_seq_slot=0, is_dummy=False,
                                 state=LlmRequestState.GENERATION_IN_PROGRESS,
                                 is_last_context_chunk=False)
    resumed = SimpleNamespace(py_seq_slot=1, is_dummy=False,
                              state=LlmRequestState.CONTEXT_INIT,
                              is_last_context_chunk=True)
    chunk = SimpleNamespace(py_seq_slot=2, is_dummy=False,
                            state=LlmRequestState.CONTEXT_INIT,
                            is_last_context_chunk=False)
    dummy = SimpleNamespace(py_seq_slot=0, is_dummy=True,
                            state=LlmRequestState.GENERATION_IN_PROGRESS,
                            is_last_context_chunk=True)
    monkeypatch.setattr("tensorrt_llm._torch.speculative.interface.prefer_pinned", lambda: False)
    # Graph padding is represented here by a dummy with a real-looking slot:
    # it still cannot observe/reset that real slot or acquire an enabled flag.
    SpecMetadata._populate_watermark_params(metadata,
                                             (generation, resumed, chunk, dummy))
    assert handler.observed == [generation, resumed]
    assert handler.invalidated == [resumed]
    assert state.enabled.tolist() == [True, True, False, False]


@pytest.mark.parametrize("clip,completed", [(False, False), (True, True), (False, True)])
def test_host_delivery_invalidates_stopped_prefix_but_not_overlap(monkeypatch, clip, completed):
    handler = Handler()
    request = SimpleNamespace(state=LlmRequestState.GENERATION_IN_PROGRESS,
                               py_seq_slot=0, tokens=[])
    host = SimpleNamespace(new_tokens=torch.tensor([[[10]], [[11]], [[12]]]),
                            new_tokens_lens=torch.tensor([3]),
                            next_draft_tokens=torch.zeros((1, 2), dtype=torch.int32),
                            verify_lens=None)
    state = SampleStateSpec(
        requests=[request], host=host, runtime_draft_len=2, draft_lens=[2],
        sampler_event=SimpleNamespace(synchronize=lambda: None),
        watermark_state=SimpleNamespace(handler=handler))
    sampler = SimpleNamespace(max_accepted_path_len=3, max_seq_len=100, draft_len=2,
                               _request_common_handling=lambda *args: None)

    def append(req, tokens, *, beam_idx, step):
        token = tokens[step][req.py_seq_slot][beam_idx]
        req.tokens.append(token)
        return token

    def stop(req, token, **kwargs):
        if completed and (clip or len(req.tokens) == 3):
            req.state = LlmRequestState.GENERATION_COMPLETE
            return True
        return False

    monkeypatch.setattr(spec_sampler_base, "add_token", append)
    monkeypatch.setattr(spec_sampler_base, "handle_stop_criteria", stop)
    SpecSampler.update_requests(sampler, state)
    assert request.tokens == ([10] if clip else [10, 11, 12])
    assert handler.invalidated == ([request] if completed else [])


def test_already_cancelled_request_invalidates_without_appending():
    handler = Handler()
    request = SimpleNamespace(state=LlmRequestState.GENERATION_COMPLETE,
                               py_seq_slot=0)
    host = SimpleNamespace(new_tokens=torch.zeros((3, 1, 1), dtype=torch.int32),
                            new_tokens_lens=torch.tensor([3]),
                            next_draft_tokens=torch.zeros((1, 2), dtype=torch.int32),
                            verify_lens=None)
    state = SampleStateSpec(
        requests=[request], host=host, runtime_draft_len=2,
        sampler_event=SimpleNamespace(synchronize=lambda: None),
        watermark_state=SimpleNamespace(handler=handler))
    SpecSampler.update_requests(SimpleNamespace(draft_len=2), state)
    assert handler.invalidated == [request]
