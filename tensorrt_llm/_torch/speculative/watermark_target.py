# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conditional keyed target law for full-coverage speculative verification."""

from collections.abc import Callable

import torch


def conditional_watermark_target(
    probs: torch.Tensor,
    seeds: torch.Tensor,
    active: torch.Tensor,
    sample_keyed: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return the target law conditioned on the watermark key and causal prefix.

    Args:
        probs: Float32 ``[B, K + 1, V]`` filtered target probabilities.
        seeds: Int64 ``[B, K + 1]`` context seeds along the proposed prefix.
        active: Bool ``[B, K + 1]`` eligibility, including request opt-out.
        sample_keyed: Keyed sampler over flattened ``[B * (K + 1), V]`` rows.

    Eligible rows become point masses at their keyed Gumbel draw. Other rows
    retain their original law. The proposal distribution is unchanged, and
    rejection sampling operates on this conditional target directly. No
    post-verification redraw is valid or necessary for eligible rows.
    """
    if probs.ndim != 3 or seeds.shape != probs.shape[:2] or active.shape != seeds.shape:
        raise ValueError("watermark target rows must bind the verification window")
    flat = probs.reshape(-1, probs.shape[-1])
    keyed = sample_keyed(flat, seeds.reshape(-1), active.reshape(-1))
    # The keyed sampler returns -1 for inactive rows; their point masses are
    # discarded below, but scatter still requires an in-range index.
    marked = torch.zeros_like(flat).scatter_(1, keyed.long().clamp_min(0)[:, None], 1.0)
    return torch.where(active[..., None], marked.reshape_as(probs), probs)
