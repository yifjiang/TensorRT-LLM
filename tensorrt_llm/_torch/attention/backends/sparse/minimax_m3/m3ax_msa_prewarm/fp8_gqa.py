# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm dev.1's short-query sparse GQA with an FP8 KV cache."""

import time

from . import core

_STATE = {}


def families(jit, api, query_heads, kv_heads, page, fused_projection):
    """Use the installed dispatch and sparse-prefill threshold for each query dtype."""
    codes = [jit._FLOAT8_E4M3FN_CODE]
    if not fused_projection:
        codes.append(jit._BFLOAT16_CODE)
    return [dict(family="dev1-fp8-kv-sparse-gqa", dtype_code=code,
                 hq=query_heads, hkv=kv_heads, sparse_mode=0, page=page,
                 split=False, q_lo=1, q_hi=api._prefill_qlen_threshold(True))
            for code in codes]


def enumerate_variants(jit, api, plans):
    """Follow mode-0's per-token sparse expansion, unlike the proxy planner."""
    variants = {}
    for family in plans:
        for q in range(family["q_lo"], family["q_hi"] + 1):
            factor = api._compute_pack_factor(q, family["hq"], family["hkv"])
            expanded, _, _, _ = api._expand_for_per_token_sparse(
                [q], [q], [0], family["page"], factor)
            max_q = max(expanded)
            args = (family["dtype_code"], 128 if max_q <= 128 else 256,
                    max_q <= 64, 0, family["page"], False, factor)
            name, params = jit._variant_key_from_runtime(*args)
            variants.setdefault(name, (args, params, family["family"], q))
    return variants


def _variants(runner, manager):
    from tensorrt_llm.bindings import DataType

    if manager.dtype != DataType.FP8:
        return None
    geometry = core.geometry(runner, manager)
    if geometry is None:
        return None
    _, jit, api = core._modules()
    config = runner.sparse_attention_config
    params = config.to_sparse_metadata_params(
        pretrained_config=runner.model.model_config.pretrained_config)
    query_heads, kv_heads = params.sharded_head_counts(runner.mapping)
    plans = families(jit, api, query_heads, kv_heads, geometry["page"],
                     bool(config.fuse_qkv_index_projection))
    variants = enumerate_variants(jit, api, plans)
    return jit, variants, geometry


def prewarm_for_runner(runner, manager):
    """Compile the stock GQA variants without loading or running them."""
    _STATE.clear()
    started = time.monotonic()
    try:
        selection = _variants(runner, manager)
        if selection is None:
            return
        jit, variants, geometry = selection
        for name, entry in variants.items():
            core._REACH.setdefault(name, entry[2])
        results, threads = core.compile_variants(jit, variants)
        failed = [name for name in variants if results[name][0].startswith("failed")]
        missing = [name for name in variants if not core._so(jit, name).exists()]
        _STATE.update(rank=geometry["rank"], variants=sorted(variants),
                      failed=failed, missing=missing)
        core.log("dev.1 FP8 sparse-GQA prewarm done rank=%d reachable=%d failed=%d "
                 "missing_after=%d seconds=%.1f threads=%d variants=%s" % (
                     geometry["rank"], len(variants), len(failed), len(missing),
                     time.monotonic() - started, threads, ",".join(sorted(variants))))
    except Exception as error:  # noqa: BLE001 -- strict admission below decides whether to refuse.
        _STATE["error"] = type(error).__name__ + ": " + str(error)
        core.log("dev.1 FP8 sparse-GQA prewarm failed: " + _STATE["error"], warn=True)


def require_complete(runner, manager):
    """Refuse readiness when any configured FP8 sparse-GQA key was not warmed."""
    selection = _variants(runner, manager)
    if selection is None:
        return
    jit, variants, _ = selection
    missing = [name for name in variants if not core._so(jit, name).exists()]
    if not _STATE or _STATE.get("error") or _STATE.get("failed") or missing:
        raise RuntimeError("FP8 sparse-GQA prewarm incomplete before readiness: "
                           f"missing={missing}, result={_STATE}")
