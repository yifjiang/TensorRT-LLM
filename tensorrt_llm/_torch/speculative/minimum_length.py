# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Minimum-length EOS bans for the full-logits, linear DSpark verifier.

The target distribution is masked before greedy/advanced/rejection sampling.
The proposal distribution is unchanged. Per-slot remaining lengths advance on
accepted device output, so an overlapped host token history cannot lag the ban.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._utils import prefer_pinned

if TYPE_CHECKING:
    from ..pyexecutor.llm_request import LlmRequest
    from .interface import SpecMetadata


def request_parameters(
    request: LlmRequest,
) -> tuple[tuple[int | None, int], int, int, int, bool]:
    """Read request identity, EOS settings, and its first resident host prefix.

    Args:
        request: Resident request with optional client ID and a request ID.

    Returns:
        Owner IDs, minimum length, original EOS, generated length, and whether
        this request commits generated tokens on the current step.
    """
    owner = (request.py_client_id, request.py_request_id)
    if owner[1] is None:
        raise RuntimeError("minimum-length state requires a unique request identity")
    minimum = max(int(getattr(request, "py_min_length", 0) or 0), 0)
    eos = getattr(request, "py_original_end_id", request.py_end_id)
    eos = -1 if eos is None else int(eos)
    generated = max(int(request.get_num_tokens(0)) - int(request.py_orig_prompt_len), 0)
    # This is the same context-chunk property the speculative sampler uses to
    # exclude chunk-only context requests from SampleStateSpec.requests.
    commit = not (
        getattr(request, "is_context_init_state", False)
        and not getattr(request, "is_last_context_chunk", True)
    )
    return owner, minimum, eos, generated, commit


def request_slot(request: LlmRequest, scratch: int) -> int:
    """Return the request slot, using neutral scratch for recognized dummies.

    Args:
        request: Resident request or graph/attention dummy.
        scratch: Reserved neutral slot after the live sequence slots.
    """
    if request.py_seq_slot is None or getattr(request, "is_dummy", False):
        return scratch
    return request.py_seq_slot


def minimum_remaining(minimum: int, generated: int) -> int:
    """Return the nonnegative number of remaining EOS-banned positions.

    Args:
        minimum: Requested minimum number of generated tokens.
        generated: Number of tokens already present in the generated prefix.
    """
    return max(int(minimum) - int(generated), 0)


@dataclass
class MinimumLengthState:
    """Shared device state and weak host ownership for sequence slots.

    ``remaining`` and ``eos`` are int64 ``[live_slots + 1]`` tensors;
    ``commit`` has the same shape and boolean dtype. The final row is neutral
    scratch. Host ownership distinguishes slot reuse across graph views.
    """

    remaining: torch.Tensor
    eos: torch.Tensor
    commit: torch.Tensor
    owners: list[weakref.ReferenceType[LlmRequest] | None] = field(default_factory=list)
    parameters: list[tuple[tuple[int | None, int], int, int] | None] = field(default_factory=list)

    @property
    def slots(self) -> int:
        """Return the number of live sequence slots, excluding scratch."""
        return self.remaining.shape[0] - 1

    @classmethod
    def create(cls: type[MinimumLengthState], slots: int) -> MinimumLengthState:
        """Allocate neutral device state for live rows and scratch.

        Args:
            slots: Number of live sequence slots, excluding scratch.
        """
        return cls(
            torch.zeros(slots + 1, dtype=torch.int64, device="cuda"),
            torch.full((slots + 1,), -1, dtype=torch.int64, device="cuda"),
            torch.zeros(slots + 1, dtype=torch.bool, device="cuda"),
            [None] * slots,
            [None] * slots,
        )


def enabled(metadata: SpecMetadata) -> bool:
    """Return whether metadata selects the supported DSpark verifier.

    Args:
        metadata: Speculative decoding metadata with its configured mode.
    """
    return metadata.spec_dec_mode.is_dspark()


def prepare(metadata: SpecMetadata) -> None:
    """Prepare shared device buffers before graph metadata is copied.

    Args:
        metadata: Parent or graph view containing the sequence-slot capacity.
    """
    if not enabled(metadata):
        return
    capacity = metadata.num_seq_slots or metadata.max_num_requests
    if capacity <= 0:
        raise RuntimeError("DSpark minimum-length state has no sequence slots")
    if metadata.minimum_length_state is None:
        metadata.minimum_length_state = MinimumLengthState.create(capacity)
    state = metadata.minimum_length_state
    if capacity > state.slots:
        raise RuntimeError("minimum-length slot storage cannot grow after allocation")
    # Independent of rejection/penalty configuration: frozen greedy DSpark has
    # both disabled but still needs the exact request-to-slot map.
    metadata.dummy_slot_row = state.slots
    if metadata.batch_slot_ids is None:
        metadata.batch_slot_ids = torch.empty(
            metadata.max_num_requests, dtype=torch.long, device="cuda"
        )
    if metadata.batch_slot_ids.numel() < metadata.max_num_requests:
        raise RuntimeError("minimum-length batch slot map is undersized")


def _device(values: list[int] | list[bool], dtype: torch.dtype) -> torch.Tensor:
    """Copy a one-dimensional host parameter list to the current CUDA device."""
    return torch.tensor(values, dtype=dtype, pin_memory=prefer_pinned()).to(
        "cuda", non_blocking=True
    )


def populate(metadata: SpecMetadata, requests: list[LlmRequest]) -> None:
    """Upload live request parameters without replacing device progress.

    Args:
        metadata: Prepared parent or graph metadata sharing minimum-length state.
        requests: Current batch in the order used by its request-to-slot map.
    """
    if not enabled(metadata):
        return
    state = metadata.minimum_length_state
    if state is None:
        raise RuntimeError("minimum-length buffers were not prepared")
    slots, commits, reset_slots, remaining, eos_values, owners = [], [], [], [], [], []
    for request in requests:
        slot = request.py_seq_slot
        if slot is None or getattr(request, "is_dummy", False):
            continue  # includes real warmup dummies with a valid sequence slot
        if not 0 <= slot < state.slots or slot in slots:
            raise RuntimeError("invalid or duplicate minimum-length sequence slot")
        owner, minimum, eos, generated, commit = request_parameters(request)
        if eos >= metadata.vocab_size:
            raise RuntimeError("minimum-length EOS is outside the target vocabulary")
        slots.append(slot)
        commits.append(commit)
        previous = state.owners[slot]
        if previous is None or previous() is not request or state.parameters[slot][0] != owner:
            # Weak ownership also distinguishes a new object whose numeric IDs
            # collide, without retaining a completed request's large token history.
            owners.append((slot, weakref.ref(request), (owner, minimum, eos)))
            reset_slots.append(slot)
            remaining.append(minimum_remaining(minimum, generated))
            eos_values.append(eos)
        elif state.parameters[slot] != (owner, minimum, eos):
            raise RuntimeError("minimum-length parameters changed within one request")
    if reset_slots:
        ids = _device(reset_slots, torch.long)
        state.remaining.index_copy_(0, ids, _device(remaining, torch.int64))
        state.eos.index_copy_(0, ids, _device(eos_values, torch.int64))
    if slots:
        state.commit.index_copy_(0, _device(slots, torch.long), _device(commits, torch.bool))
    # Publish ownership only after every upload has been enqueued successfully.
    # A synchronous allocation/upload error must make the next call reset again.
    for slot, owner_ref, parameters in owners:
        state.owners[slot] = owner_ref
        state.parameters[slot] = parameters


def apply(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    metadata: SpecMetadata,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Mask target EOS scores and return their bounded restoration record.

    Args:
        logits: Floating-point full-vocabulary target scores shaped
            ``[contexts + verified_generation_rows, vocab]``.
        draft_tokens: Integer proposal IDs shaped ``[generations, draft_len]``.
        num_contexts: Number of leading one-row context requests.
        batch_size: Total number of context and generation requests.
        metadata: Prepared metadata with uniform or packed ragged row mapping.

    Returns:
        The same logits tensor and an optional tuple of row indices, EOS column
        indices, and original scores. Restore it after target sampling.
    """
    if not enabled(metadata):
        return logits, None
    state = metadata.minimum_length_state
    if state is None or metadata.batch_slot_ids is None:
        raise RuntimeError("DSpark minimum-length metadata is missing")
    if logits.dim() != 2 or logits.shape[1] != metadata.vocab_size:
        raise RuntimeError("DSpark minimum-length needs normalized full target logits")
    num_gens = batch_size - num_contexts
    batch_slots = metadata.batch_slot_ids[:batch_size]
    if metadata.is_ragged_verify:
        from .ragged_helpers import row_ids_from_lens

        lens, starts, total = (
            metadata.verify_lens,
            metadata.qo_indptr,
            metadata.total_verify_tokens,
        )
        if lens is None or starts is None or total is None:
            raise RuntimeError("minimum-length ragged verification is incomplete")
        if lens.numel() != num_gens or starts.numel() != num_gens + 1:
            raise RuntimeError("minimum-length ragged verification shape differs")
        request_ids = row_ids_from_lens(lens.to(torch.int64), total=int(total))
        positions = torch.arange(int(total), device=logits.device) - starts[request_ids]
    else:
        width = draft_tokens.shape[1] + 1
        flat = torch.arange(num_gens * width, device=logits.device)
        request_ids, positions = flat // width, flat % width
    row_slots = torch.cat((batch_slots[:num_contexts], batch_slots[num_contexts:][request_ids]))
    positions = torch.cat(
        (torch.zeros(num_contexts, dtype=torch.long, device=logits.device), positions)
    )
    if row_slots.numel() != logits.shape[0]:
        raise RuntimeError("minimum-length normalized target row count differs")
    eos = state.eos[row_slots]
    banned = (positions < state.remaining[row_slots]) & (eos >= 0)
    # Save only one original EOS score per row. The caller restores it after
    # every target sampler/probability read has been enqueued on this stream,
    # before returning raw logits or starting the next draft forward. This is
    # a bounded in-place transaction, not a full-vocabulary clone per graph.
    rows = torch.arange(logits.shape[0], device=logits.device)
    columns = eos.clamp(min=0, max=logits.shape[1] - 1)
    original = logits[rows, columns].clone()
    logits[rows, columns] = torch.where(banned, torch.full_like(original, -float("inf")), original)
    return logits, (rows, columns, original)


def restore(
    logits: torch.Tensor, saved: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
) -> None:
    """Restore EOS scores after target consumers on the same CUDA stream.

    Args:
        logits: The floating-point target score tensor modified by ``apply``.
        saved: Row indices, EOS column indices, and original scores, or ``None``.
    """
    if saved is not None:
        rows, columns, original = saved
        logits[rows, columns] = original


def commit(
    accepted: tuple[torch.Tensor, torch.Tensor], batch_size: int, metadata: SpecMetadata
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance device progress by the actual accepted prefix.

    Args:
        accepted: Integer token IDs ``[batch, accepted_width]`` and accepted
            counts ``[batch]`` returned by target verification or sampling.
        batch_size: Number of active request-to-slot entries.
        metadata: Metadata sharing the device remaining-length state.

    Returns:
        The original accepted-token and count tensors.
    """
    if not enabled(metadata):
        return accepted
    state = metadata.minimum_length_state
    slots = metadata.batch_slot_ids[:batch_size]
    _, counts = accepted
    advance = counts.to(torch.int64) * state.commit[slots].to(torch.int64)
    state.remaining.index_add_(0, slots, -advance)
    state.remaining.clamp_(min=0)
    return accepted
