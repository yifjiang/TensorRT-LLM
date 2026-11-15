# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real keyed kernels composed with block verification and committed state."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.watermark import WatermarkHandler
from tensorrt_llm._torch.speculative.block_verification import block_verify, counter_uniforms
from tensorrt_llm._torch.speculative.watermark_target import conditional_watermark_target
from tensorrt_llm.watermark import WatermarkConfig
from tensorrt_llm.watermark.sampling import gumbel_sample_from_probs, gumbel_sample_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dedupe", ["none", "single_turn", "all"])
@pytest.mark.parametrize("capture", [False, True])
def test_full_block_keyed_prefix_commit(dedupe, capture):
    config = WatermarkConfig(key=7, deduplicate_contexts=dedupe)
    handler = WatermarkHandler(config, max_num_sequences=4, max_seq_len=64)
    requests = [
        SimpleNamespace(
            py_request_id=i,
            py_seq_slot=i,
            py_watermarking=None,
            get_tokens=lambda beam: [1, 2, 3, 4],
        )
        for i in range(2)
    ]
    handler.observe(requests, lambda request: True)
    slots = torch.tensor([0, 1], device="cuda")
    enabled = torch.tensor([True, False], device="cuda")
    drafts = torch.tensor([[2, 3, 4], [4, 3, 2]], dtype=torch.int32, device="cuda")
    # Zero-probability entries model grammar exclusions in the target law.
    probs = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], device="cuda").expand(2, 4, 5).contiguous()
    proposal = torch.full((2, 3, 5), 0.2, device="cuda")
    seed = torch.tensor([33, 55], device="cuda")
    offset = torch.tensor([1024, 4096], device="cuda")
    original = probs.clone()

    def run():
        plan = handler.plan(slots, enabled, drafts)
        p = conditional_watermark_target(probs, plan.seeds, plan.active, gumbel_sample_from_probs)
        tokens, counts = block_verify(
            drafts,
            proposal,
            p,
            counter_uniforms(seed, offset, 3),
            lambda weights: torch.multinomial(weights, 1).squeeze(-1),
        )
        return plan, p, tokens, counts

    # Make at least one real draft equal the causal keyed target. This is a
    # component boundary test; model qualification must use genuine drafts.
    plan = handler.plan(slots, enabled, drafts)
    first = gumbel_sample_reference(probs[:, 0], plan.seeds[:, 0])
    drafts[:, 0] = first
    if capture:
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan, p, tokens, counts = run()
        graph.replay()
    else:
        plan, p, tokens, counts = run()
    reference = gumbel_sample_reference(probs.reshape(-1, 5), plan.seeds.reshape(-1)).reshape(2, 4)
    keyed_rows = p.argmax(-1)
    assert torch.equal(keyed_rows[plan.active], reference[plan.active])
    assert torch.equal(p[~plan.active], probs[~plan.active])
    assert torch.equal(probs, original)
    if plan.active[0, 0]:
        assert counts[0].item() >= 2
    for i in range(2):
        for j in range(counts[i].item()):
            token = tokens[i, j].item()
            assert token != 0
            if plan.active[i, j]:
                assert token == reference[i, j].item()
    handler.commit(plan, tokens, counts)
    assert handler._count[0].item() == counts[0].item()
    assert handler._count[1].item() == 0
