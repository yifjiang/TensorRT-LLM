# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require complete native MSA warmup, including short-query FP8 sparse GQA."""

import contextlib
import os

from . import core, fp8_gqa, hook, sdwarm


def _require_complete():
    return os.environ.get("M31_PREWARM_REQUIRE_COMPLETE", "1").strip().lower() not in hook.OFF


def _require_fmha(runner, kvm):
    geo = core.geometry(runner, kvm)
    _, jit, api = core._modules()
    code = jit._dlpack_dtype_code(geo["idx_dtype"])
    reach = core.enumerate_variants(jit, api, core.families_for(geo["hidx"], code, geo["page"]))
    missing = [name for name in reach if not core._so(jit, name).exists()]
    plan_ready = (jit.CACHE_BASE / "plan" / "fmha_sm100_plan.so").exists()
    if missing or not plan_ready:
        raise RuntimeError(
            "MSA prewarm incomplete before readiness: "
            f"missing_variants={missing}, plan_ready={plan_ready}"
        )
    fp8_gqa.require_complete(runner, kvm)


@contextlib.contextmanager
def warmup_scope(runner, kvm):
    """Warm the installed kernels and verify completion before readiness."""
    if not hook.ENABLED or core.geometry(runner, kvm) is None:
        yield
        return
    sdwarm._S["last"] = None
    with hook.warmup_scope(runner):
        fp8_gqa.prewarm_for_runner(runner, kvm)
        if _require_complete():
            fp8_gqa.require_complete(runner, kvm)
        yield
    if _require_complete():
        _require_fmha(runner, kvm)
    if _require_complete() and sdwarm.enabled():
        result = sdwarm._S.get("last")
        expected_batch = int(runner._config.max_batch_size)
        if (not result or result["failed"] or not result["calls"]
                or result.get("max_batch") != expected_batch):
            raise RuntimeError(
                "MSA sparse-decode prewarm incomplete before readiness: "
                f"result={result}, recorded={sdwarm._S['recorded']}, "
                f"skipped={sdwarm._S['skipped']}"
            )
    core.log("branch prewarm verified before readiness")
