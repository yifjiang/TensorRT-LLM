# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive conditional-watermark/block-verifier oracle on CPU."""

import itertools
import math

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative import block_verification as block
from tensorrt_llm._torch.speculative import watermark_target as marked
from tensorrt_llm.watermark import prf


def keyed_numpy(probs, seeds, active):
    results = []
    for p, seed, enabled in zip(probs.numpy(), seeds.tolist(), active.tolist()):
        if not enabled:
            results.append(-1)
            continue
        u = prf.philox_uniform(seed & (2**64 - 1), np.arange(len(p), dtype=np.uint64))
        score = np.divide(-np.log(u), p, out=np.full(len(p), np.inf), where=p > 0)
        results.append(int(score.argmin()))
    return torch.tensor(results, dtype=torch.int32)


def law(prefix, enabled=True):
    p = np.array([0.3, 0.7]) if not prefix else np.array([0.65, 0.35])
    # Repeated causal contexts deliberately turn the key off at some positions.
    eligible = enabled and not (len(prefix) > 1 and prefix[-1] == prefix[-2])
    seed = prf.context_seed(7, list(prefix[-2:]))
    if eligible:
        t = keyed_numpy(
            torch.tensor(p[None], dtype=torch.float32),
            torch.tensor([seed if seed < 2**63 else seed - 2**64]),
            torch.tensor([True]),
        ).item()
        p = np.eye(2)[t]
    return p, seed, eligible


def proposal(prefix):
    # An ancestral proposal whose law depends on the actual preceding token.
    return np.array([0.2, 0.8]) if not prefix or prefix[-1] == 0 else np.array([0.8, 0.2])


@pytest.mark.parametrize("enabled", [False, True])
def test_exhaustive_ancestral_block_law(enabled):
    """Enumerate drafts, accept coins, and residuals; continue to a fixed horizon.

    The independent scalar block oracle computes each coin interval and the
    probability of its cell. The production verifier is evaluated inside every
    cell. Its variable prefix, continued with the conditional AR law, must equal
    the independently enumerated three-token AR distribution exactly.
    """
    horizon, k = 3, 2
    actual = {t: 0.0 for t in itertools.product(range(2), repeat=horizon)}
    expected = {}
    for seq in actual:
        probability = 1.0
        for j, t in enumerate(seq):
            probability *= law(seq[:j], enabled)[0][t]
        expected[seq] = probability
    accepted_drafts = 0.0
    for draft in itertools.product(range(2), repeat=k):
        q = np.array([proposal(draft[:j]) for j in range(k)])
        raw_p = np.array([([0.3, 0.7] if j == 0 else [0.65, 0.35]) for j in range(k + 1)])
        info = [law(draft[:j], enabled) for j in range(k + 1)]
        seeds = torch.tensor([[s if s < 2**63 else s - 2**64 for _, s, _ in info]])
        active = torch.tensor([[a for _, _, a in info]])
        target = marked.conditional_watermark_target(
            torch.tensor(raw_p[None], dtype=torch.float32), seeds, active, keyed_numpy
        )
        p = np.array([v[0] for v in info])
        np.testing.assert_allclose(target.numpy()[0], p, atol=3e-8)
        proposal_mass = math.prod(q[j, t] for j, t in enumerate(draft))
        w, running = [], 1.0
        for j, t in enumerate(draft):
            running = min(1.0, running * p[j, t] / q[j, t])
            w.append(running)
        h = w.copy()
        r = np.maximum(w[0] * p[1] - q[1], 0).sum()
        h[0] = r / (r + 1 - w[0]) if r + 1 - w[0] else 1.0
        for coins in itertools.product([False, True], repeat=k):
            mass = proposal_mass * math.prod(h[j] if c else 1 - h[j] for j, c in enumerate(coins))
            if mass == 0:
                continue
            u = [[h[j] / 2 if c else (1 + h[j]) / 2 for j, c in enumerate(coins)]]
            length = max([j + 1 for j, c in enumerate(coins) if c] or [0])
            residual = (
                p[k]
                if length == k
                else np.maximum((w[length - 1] if length else 1) * p[length] - q[length], 0)
            )
            residual = residual / residual.sum()
            for emit in range(2):
                if residual[emit] == 0:
                    continue

                def sample(weights):
                    np.testing.assert_allclose(
                        (weights / weights.sum(-1, keepdim=True)).numpy()[0], residual, atol=1e-6
                    )
                    return torch.tensor([emit])

                out, count = block.block_verify(
                    torch.tensor([draft]),
                    torch.tensor(q[None], dtype=torch.float32),
                    target,
                    torch.tensor(u),
                    sample,
                )
                assert count.item() == length + 1
                prefix = tuple(out[0, : count.item()].tolist())
                assert prefix == draft[:length] + (emit,)
                accepted_drafts += mass * residual[emit] * length
                for tail in itertools.product(range(2), repeat=horizon - len(prefix)):
                    seq = prefix + tail
                    tail_mass = math.prod(
                        law(seq[:j], enabled)[0][seq[j]] for j in range(len(prefix), horizon)
                    )
                    actual[seq] += mass * residual[emit] * tail_mass
    assert accepted_drafts > 0
    for seq in actual:
        assert actual[seq] == pytest.approx(expected[seq], abs=2e-7)
    assert sum(actual.values()) == pytest.approx(1.0)


def test_optout_and_probability_storage_are_unchanged():
    p = torch.tensor([[[0.1, 0.9], [0.8, 0.2]], [[0.7, 0.3], [0.2, 0.8]]])
    original = p.clone()
    seeds = torch.tensor([[5, 6], [7, 8]])
    active = torch.tensor([[True, False], [False, True]])
    got = marked.conditional_watermark_target(p, seeds, active, keyed_numpy)
    torch.testing.assert_close(got[~active], p[~active], rtol=0, atol=0)
    assert torch.equal(got[active].sum(-1), torch.ones(2))
    assert torch.all((got[active] == 0) | (got[active] == 1))
    torch.testing.assert_close(p, original, rtol=0, atol=0)
    order = torch.tensor([1, 0])
    shuffled = marked.conditional_watermark_target(
        p[order], seeds[order], active[order], keyed_numpy
    )
    torch.testing.assert_close(shuffled, got[order], rtol=0, atol=0)


def test_counter_state_follows_requests_through_batch_reordering():
    seed = torch.tensor([33, 55, 77])
    offset = torch.tensor([1024, 4096, 8192])
    u = block.counter_uniforms(seed, offset, 7)
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        block.counter_uniforms(seed[order], offset[order], 7), u[order], rtol=0, atol=0
    )
    assert torch.all((u > 0) & (u < 1))
    assert not torch.equal(u, block.counter_uniforms(seed, offset + 256, 7))
