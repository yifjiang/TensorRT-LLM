# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native sampler stream preservation for filtered watermark opt-outs."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.sampler.ops.custom import (
    fused_sample_from_logits,
    fused_sample_from_logits_with_probs,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.watermark import WatermarkConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_NUM_ROWS = 4
_VOCAB_SIZE = 151936
_MIXED_FLAGS = (True, False, True, False)


@pytest.fixture
def filtered_logits() -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(20261110)
    return (
        torch.randn(
            (_NUM_ROWS, _VOCAB_SIZE), generator=generator, device="cuda", dtype=torch.float32
        )
        * 0.2
    )


def _make_sampler(
    config: WatermarkConfig | None, flags: tuple[bool, ...]
) -> tuple[TorchSampler, list[LlmRequest]]:
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=128,
            max_draft_len=0,
            max_num_sequences=_NUM_ROWS,
            max_beam_width=1,
            max_total_draft_tokens=0,
            watermark_config=config,
        )
    )
    requests = []
    for slot, flag in enumerate(flags):
        params = SamplingParams(seed=431 + slot, temperature=0.9, top_k=100, top_p=0.98, min_p=0.01)
        requests.append(
            LlmRequest(
                request_id=slot,
                max_new_tokens=64,
                input_tokens=[10, 10, 10, 10],
                sampling_config=SamplingConfig(params._get_sampling_config()),
                seq_slot=slot,
                is_streaming=False,
                watermarking=flag,
            )
        )
    slots = torch.arange(_NUM_ROWS, device="cuda", dtype=torch.int64)
    sampler._slot_params.admit(requests, slots)
    sampler._seed_manager.observe(requests)
    if sampler._watermark is not None:
        sampler._watermark.observe(requests, is_neutral=sampler._watermark_row_is_neutral)
    return sampler, requests


@pytest.mark.parametrize("return_probs", [False, True])
def test_filtered_mixed_opt_out_preserves_native_seeded_stream(filtered_logits, return_probs):
    stock, stock_requests = _make_sampler(None, (False,) * _NUM_ROWS)
    marked, marked_requests = _make_sampler(WatermarkConfig(key=99), _MIXED_FLAGS)
    slots = list(range(_NUM_ROWS))
    for step in range(32):
        expected, expected_probs, _ = stock._sample_fused(
            filtered_logits, slots, return_probs=return_probs
        )
        actual, actual_probs, _ = marked._sample_fused(
            filtered_logits, slots, return_probs=return_probs
        )
        torch.testing.assert_close(
            actual[[1, 3]],
            expected[[1, 3]],
            rtol=0,
            atol=0,
            msg=f"Opt-out stream changed at step {step}",
        )
        expected_offsets = torch.full((_NUM_ROWS,), step + 1, dtype=torch.int64)
        assert torch.equal(stock._seed_manager._offsets, expected_offsets)
        assert torch.equal(marked._seed_manager._offsets, expected_offsets)
        if return_probs:
            torch.testing.assert_close(actual_probs, expected_probs)
        else:
            assert actual_probs is expected_probs is None
    # Keep the real request objects alive for the handlers' weak ownership references.
    assert len(stock_requests) == len(marked_requests) == _NUM_ROWS


@pytest.mark.parametrize("return_probs", [False, True])
@pytest.mark.parametrize("all_enabled", [False, True])
def test_filtered_repeat_fallback_preserves_marked_native_draw(
    filtered_logits, return_probs, all_enabled, record_property
):
    flags = (True,) * _NUM_ROWS if all_enabled else _MIXED_FLAGS
    sampler, requests = _make_sampler(WatermarkConfig(key=99), flags)
    handler = sampler._watermark
    assert handler is not None
    slots = torch.arange(_NUM_ROWS, device="cuda", dtype=torch.int64)
    enabled = torch.tensor(flags, device="cuda", dtype=torch.bool)
    # Committing token 10 keeps the tail unchanged, so the next marked
    # position repeats its context. Prior draws advance the native stream too.
    for _ in range(3):
        rows = handler.plan(slots, enabled)
        handler.commit(rows, torch.full((_NUM_ROWS,), 10, device="cuda", dtype=torch.int32))
        sampler._seed_manager.take_row_seeds(list(range(_NUM_ROWS)), device=filtered_logits.device)
    next_rows = handler.plan(slots, enabled)
    assert not next_rows.active.any().item()
    assert torch.equal(
        sampler._seed_manager._offsets, torch.full((_NUM_ROWS,), 3, dtype=torch.int64)
    )

    temperature, top_k, top_p, min_p = sampler._slot_params.gather(slots)
    seeds = torch.arange(431, 435, device="cuda", dtype=torch.int64)
    offsets = torch.full((_NUM_ROWS,), 3, device="cuda", dtype=torch.int64)
    native_args = (filtered_logits, temperature, top_k, top_p, min_p)
    paired_tokens, paired_probs = fused_sample_from_logits_with_probs(
        *native_args, seed=seeds, offset=offsets
    )
    stock_tokens = fused_sample_from_logits(*native_args, seed=seeds, offset=offsets)
    record_property("native_oracle_offset", 3)
    record_property("native_tokens_only", str(stock_tokens.cpu().tolist()))
    record_property("native_with_probs", str(paired_tokens.cpu().tolist()))
    record_property(
        "native_oracle_distinct_rows",
        str((stock_tokens != paired_tokens).nonzero().flatten().cpu().tolist()),
    )
    actual, actual_probs, _ = sampler._sample_fused(
        filtered_logits, list(range(_NUM_ROWS)), return_probs=return_probs
    )
    torch.testing.assert_close(
        actual[enabled],
        paired_tokens[enabled],
        rtol=0,
        atol=0,
        msg="Marked repeated-context fallback changed its native draw",
    )
    if return_probs:
        torch.testing.assert_close(actual[~enabled], paired_tokens[~enabled], rtol=0, atol=0)
        torch.testing.assert_close(actual_probs, paired_probs)
    else:
        torch.testing.assert_close(
            actual[~enabled],
            stock_tokens[~enabled],
            rtol=0,
            atol=0,
            msg="Opt-out repeat fallback changed the native seeded stream",
        )
        assert actual_probs is None
    assert torch.equal(
        sampler._seed_manager._offsets, torch.full((_NUM_ROWS,), 4, dtype=torch.int64)
    )
    assert len(requests) == _NUM_ROWS
