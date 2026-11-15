# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ragged target-only recovery must score the true target bonus distribution."""

import pytest
import torch

from tensorrt_llm._torch.speculative.watermark_ragged import ragged_recovery_inputs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_local_bonus_uses_target_not_artificial_padding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Row zero has one real proposal and a local bonus at position one;
    # row one executes the full three-proposal window.
    proposals = torch.tensor(
        [[[.4, .6], [1., 0.], [1., 0.]],
         [[.2, .8], [.3, .7], [.4, .6]]], device=device)
    original = proposals.clone()
    counts, recovery = ragged_recovery_inputs(
        torch.tensor([4, 4], dtype=torch.int32, device=device), proposals,
        torch.tensor([2, 4], dtype=torch.int32, device=device))
    assert counts.tolist() == [2, 4]
    target_bonus = torch.tensor([.7, .3], device=device)
    residual = (target_bonus - recovery[0, 1]).clamp(min=0)
    torch.testing.assert_close(residual / residual.sum(), target_bonus)
    # Subtracting the artificial native proposal would incorrectly ban token0.
    wrong = (target_bonus - original[0, 1]).clamp(min=0)
    assert wrong[0] == 0 and residual[0] > 0
    torch.testing.assert_close(recovery[1], original[1])
    torch.testing.assert_close(proposals, original)


def test_first_rejection_keeps_real_proposal_distribution():
    proposals = torch.tensor([[[.8, .2], [1., 0.], [1., 0.]]])
    counts, recovery = ragged_recovery_inputs(
        torch.tensor([1], dtype=torch.int32), proposals,
        torch.tensor([2], dtype=torch.int32))
    target = torch.tensor([.3, .7])
    residual = (target - recovery[0, counts[0] - 1]).clamp(min=0)
    torch.testing.assert_close(residual / residual.sum(), torch.tensor([0., 1.]))


@pytest.mark.parametrize("window", [1, 2, 4])
def test_reorder_zero_draft_and_padding_never_grow_executed_prefix(window):
    proposals = torch.full((3, 3, 2), .5)
    lengths = torch.tensor([window, 4, 1], dtype=torch.int32)
    native = torch.tensor([4, 2, 3], dtype=torch.int32)
    order = torch.tensor([2, 0, 1])
    counts, recovery = ragged_recovery_inputs(native, proposals, lengths)
    reordered, reordered_probs = ragged_recovery_inputs(
        native[order], proposals[order], lengths[order])
    torch.testing.assert_close(reordered, counts[order])
    torch.testing.assert_close(reordered_probs, recovery[order])
    assert counts[2] == 1
    assert recovery[2].count_nonzero() == 0


def test_incompatible_request_rows_are_rejected():
    with pytest.raises(ValueError, match="one window"):
        ragged_recovery_inputs(torch.tensor([1, 2]), torch.zeros(1, 3, 2),
                               torch.tensor([1, 2]))
