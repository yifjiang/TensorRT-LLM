# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose real native ragged acceptance with the PR's actual keyed recovery."""

import json
import os

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.watermark_ragged import ragged_recovery_inputs
from tensorrt_llm.watermark.prf import context_seeds_torch
from tensorrt_llm.watermark.sampling import gumbel_sample_reference, keyed_recovery


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA native rejection and replay")
def test_native_ragged_target_only_recovery_full_vocabulary(tmp_path):
    assert torch.cuda.is_available(), "This composition regression requires an actual CUDA device"
    assert interface._CHAIN_SPEC_OVERRIDE is None
    assert float(os.environ.get("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", "0")) == 0
    device = torch.device("cuda")
    batch, drafts, vocab, width = 128, 3, 200064, 4
    generator = torch.Generator(device=device).manual_seed(210472026)
    # Public, synthetic P/Q. Prefix acceptance still comes from the native
    # deterministic kernel's ordinary request seeds and accept coins.
    target = torch.rand((batch, drafts + 1, vocab), generator=generator, device=device)
    target /= target.sum(dim=-1, keepdim=True)
    target *= .5
    target[..., 0] += .5
    proposal = torch.rand((batch, drafts, vocab), generator=generator, device=device)
    proposal /= proposal.sum(dim=-1, keepdim=True)
    windows = (torch.arange(batch, device=device) % (drafts + 1) + 1).to(torch.int32)
    windows[0] = drafts + 1
    # P=Q is a normal exact-distribution control for a full native window.
    proposal[0].copy_(target[0, :drafts])
    tokens = torch.multinomial(proposal.reshape(-1, vocab), 1,
                                generator=generator).reshape(batch, drafts).to(torch.int32)
    positions = torch.arange(drafts, device=device).view(1, -1)
    real_proposals = positions < (windows - 1).view(-1, 1)
    tokens = torch.where(real_proposals, tokens, torch.zeros_like(tokens))
    proposal = torch.where(real_proposals.unsqueeze(-1), proposal,
                            torch.zeros_like(proposal))
    proposal[..., 0] += (~real_proposals).to(proposal.dtype)
    target_positions = torch.arange(drafts + 1, device=device).view(1, -1)
    real_targets = target_positions < windows.view(-1, 1)
    target = torch.where(real_targets.unsqueeze(-1), target, torch.zeros_like(target))
    target[..., 0] += (~real_targets).to(target.dtype)
    active = torch.ones((batch, drafts + 1), dtype=torch.bool, device=device)
    active[-4:] = False
    windows[-4:] = 1
    tokens[-4:] = 0
    proposal[-4:] = 0
    proposal[-4:, :, 0] = 1
    target[-4:] = 0
    target[-4:, :, 0] = 1
    request_seeds = torch.arange(batch, dtype=torch.int64, device=device) + 2026111000
    request_offsets = torch.full_like(request_seeds, 512)
    tails = torch.arange(batch * width, dtype=torch.int64, device=device).reshape(batch, width) + 71
    history = torch.cat((tails, tokens.to(torch.int64)), dim=1)
    contexts = history.unfold(1, width, 1).contiguous()
    seeds = context_seeds_torch(424242, contexts.reshape(-1, width)).reshape(batch, drafts + 1)
    target_before, proposal_before, tokens_before = target.clone(), proposal.clone(), tokens.clone()

    def execute(p, q, draft_ids, lengths, rng_seeds, rng_offsets, row_seeds, enabled):
        accepted, native_counts = interface.rejection_sampling_one_model(
            draft_probs=q, draft_token_ids=draft_ids, target_probs=p,
            deterministic=True, seed=rng_seeds, offset=rng_offsets)
        original_accepted, original_counts = accepted.clone(), native_counts.clone()
        counts, real_q = ragged_recovery_inputs(native_counts, q, lengths)
        recovered = keyed_recovery(accepted, counts, p, real_q, row_seeds, enabled)
        return accepted, native_counts, counts, recovered, original_accepted, original_counts

    accepted, native_counts, counts, recovered, accepted_before, native_counts_before = execute(
        target, proposal, tokens, windows, request_seeds, request_offsets, seeds, active)
    recovery_position = counts.to(torch.int64) - 1
    row = torch.arange(batch, device=device)
    p = target[row, recovery_position]
    q = proposal[row, recovery_position.clamp(max=drafts - 1)]
    local_bonus = recovery_position == windows.to(torch.int64) - 1
    q = torch.where(local_bonus.unsqueeze(1), torch.zeros_like(q), q)
    residual = (p - q).clamp(min=0)
    residual = torch.where(residual.sum(dim=-1, keepdim=True) > 0, residual, p)
    expected = gumbel_sample_reference(residual, seeds[row, recovery_position])
    enabled = active[row, recovery_position]
    actual_last = recovered[row, recovery_position]
    torch.testing.assert_close(actual_last[enabled], expected[enabled])
    torch.testing.assert_close(actual_last[~enabled], accepted[row, recovery_position][~enabled])
    prefix = target_positions < recovery_position.unsqueeze(1)
    torch.testing.assert_close(recovered[prefix], accepted[prefix])
    real_prefix = positions < recovery_position.unsqueeze(1)
    torch.testing.assert_close(recovered[:, :drafts][real_prefix], tokens[real_prefix])
    unchanged = target_positions != recovery_position.unsqueeze(1)
    torch.testing.assert_close(recovered[unchanged], accepted[unchanged])
    torch.testing.assert_close(target, target_before, rtol=0, atol=0)
    torch.testing.assert_close(proposal, proposal_before, rtol=0, atol=0)
    torch.testing.assert_close(tokens, tokens_before)
    torch.testing.assert_close(accepted, accepted_before)
    torch.testing.assert_close(native_counts, native_counts_before)
    assert counts[0] == drafts + 1
    assert ((counts == 1) & (windows > 1) & enabled).any()
    assert ((counts > 1) & (counts < windows) & enabled).any()
    assert ((counts == windows) & (windows > 1) & enabled).any()
    assert ((counts == 1) & (windows == 1)).any()
    assert (local_bonus & enabled & (actual_last == 0)).any()

    # Per-request random states and context seeds travel with each row.
    order = torch.randperm(batch, generator=generator, device=device)
    reordered = execute(target[order], proposal[order], tokens[order], windows[order],
                         request_seeds[order], request_offsets[order], seeds[order], active[order])
    originals = (accepted, native_counts, counts, recovered, accepted_before, native_counts_before)
    for original, changed in zip(originals, reordered):
        torch.testing.assert_close(changed, original[order])

    # The ordinary native accept op and both recovery components are replayed,
    # with the same fixed inputs after their eager warmup.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        execute(target, proposal, tokens, windows, request_seeds, request_offsets, seeds, active)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = execute(target, proposal, tokens, windows,
                                 request_seeds, request_offsets, seeds, active)
    graph.replay()
    torch.cuda.synchronize()
    for original, captured in zip(originals, graph_outputs):
        torch.testing.assert_close(captured, original)
    torch.testing.assert_close(target, target_before, rtol=0, atol=0)
    torch.testing.assert_close(proposal, proposal_before, rtol=0, atol=0)
    torch.testing.assert_close(tokens, tokens_before)
    record = {"status": "PASS_REAL_NATIVE_RAGGED_PR_RECOVERY_COMPOSITION", "vocabulary": vocab,
              "batch": batch, "max_drafts": drafts, "actual_CUDA_graph_replay": True,
              "native_acceptance_override": False, "native_P_Q_and_counts_unchanged": True,
              "oracle": "Official PR Gumbel float64 reference with actual per-request residual/bonus",
              "count_histogram": torch.bincount(counts.to(torch.int64)).cpu().tolist(),
              "bonus_token0_observed": True, "reordered_rows": True, "padded_inactive_rows": 4}
    (tmp_path / "NATIVE-RAGGED-CUDA-RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
