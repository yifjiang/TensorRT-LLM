# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind keyed recovery to the executed windows of ragged rejection sampling."""

import torch


def ragged_recovery_inputs(
    num_accepted: torch.Tensor,
    draft_probs: torch.Tensor,
    verify_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return executed counts and real proposals for the keyed recovery draw.

    The native rejection kernel pads missing proposals with a point mass at
    token zero. Those proposals emulate a target bonus draw for acceptance;
    they are not draft distributions to subtract from that bonus position.

    Args:
        num_accepted: ``[B]`` native accepted counts, including its bonus.
        draft_probs: ``[B, K, V]`` padded native proposal distributions.
        verify_lens: ``[B]`` executed windows, including each request's bonus.

    Returns:
        Counts capped at the executed window and private proposal storage with
        no proposal at or after each request's local bonus. Inputs stay intact.
    """
    if (draft_probs.ndim != 3 or num_accepted.ndim != 1
            or verify_lens.shape != num_accepted.shape
            or draft_probs.shape[0] != num_accepted.shape[0]):
        raise ValueError("ragged recovery requires one window per proposal row")
    if (num_accepted.dtype not in (torch.int32, torch.int64)
            or verify_lens.dtype not in (torch.int32, torch.int64)
            or verify_lens.device != num_accepted.device
            or draft_probs.device != num_accepted.device):
        raise ValueError("ragged recovery counts and proposals must share a device")
    counts = torch.minimum(num_accepted, verify_lens.to(num_accepted.dtype))
    positions = torch.arange(draft_probs.shape[1], device=draft_probs.device)
    real_proposal = positions.unsqueeze(0) < (verify_lens - 1).unsqueeze(1)
    proposals = torch.where(real_proposal.unsqueeze(2), draft_probs,
                            torch.zeros_like(draft_probs))
    return counts, proposals
