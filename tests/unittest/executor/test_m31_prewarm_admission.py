# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU controls for the native runner's strict MSA warmup admission scope."""

import contextlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only


def _load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[3]
_PKG = _ROOT / "tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/m3ax_msa_prewarm"
_NAME = "m31_native_admission_cpu"
_load(_NAME, _PKG / "__init__.py", package=True)
core = _load(_NAME + ".core", _PKG / "core.py")
sdwarm = _load(_NAME + ".sdwarm", _PKG / "sdwarm.py")
hook = _load(_NAME + ".hook", _PKG / "hook.py")
fp8_gqa = _load(_NAME + ".fp8_gqa", _PKG / "fp8_gqa.py")
integration = _load(_NAME + ".integration", _PKG / "integration.py")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.runner = types.SimpleNamespace(_config=types.SimpleNamespace(max_batch_size=82))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(hook, "ENABLED", True))
        self.stack.enter_context(patch.object(core, "geometry", return_value={}))
        self.stack.enter_context(patch.object(core, "log"))
        self.stack.enter_context(patch.object(integration, "_require_complete", return_value=True))
        self.stack.enter_context(patch.object(integration, "_require_fmha"))
        self.stack.enter_context(patch.object(sdwarm, "enabled", return_value=True))
        self.stack.enter_context(patch.object(fp8_gqa, "prewarm_for_runner"))
        self.fp8_gate = self.stack.enter_context(patch.object(fp8_gqa, "require_complete"))
        self.stack.enter_context(patch.dict(sdwarm._S, {"last": None}, clear=False))

    def finish_with(self, result):
        return patch.object(sdwarm, "finish", side_effect=lambda *a, **kw: sdwarm._S.update(last=result))

    def test_complete_warmup_is_admitted(self):
        with self.finish_with({"failed": 0, "calls": 4, "max_batch": 82}):
            with integration.warmup_scope(self.runner, object()):
                pass

    def test_failed_or_missing_decode_coverage_is_refused(self):
        for result in (None, {"failed": 1, "calls": 4, "max_batch": 82},
                       {"failed": 0, "calls": 0, "max_batch": 82},
                       {"failed": 0, "calls": 4, "max_batch": 81}):
            with self.subTest(result=result), self.finish_with(result):
                with self.assertRaises(RuntimeError):
                    with integration.warmup_scope(self.runner, object()):
                        pass

    def test_prior_success_cannot_mask_skipped_warmup(self):
        sdwarm._S["last"] = {"failed": 0, "calls": 4, "max_batch": 82}
        with patch.object(sdwarm, "finish"), self.assertRaises(RuntimeError):
            with integration.warmup_scope(self.runner, object()):
                pass

    def test_missing_fp8_library_refuses_before_scheduled_warmup(self):
        self.fp8_gate.side_effect = RuntimeError("missing FP8 library")
        entered = False
        with patch.object(sdwarm, "finish"), self.assertRaises(RuntimeError):
            with integration.warmup_scope(self.runner, object()):
                entered = True
        self.assertFalse(entered)

    def test_native_proxy_failure_refuses_after_scheduled_warmup(self):
        integration._require_fmha.side_effect = RuntimeError("missing native proxy library")
        with self.finish_with({"failed": 0, "calls": 4, "max_batch": 82}):
            with self.assertRaises(RuntimeError):
                with integration.warmup_scope(self.runner, object()):
                    pass


if __name__ == "__main__":
    unittest.main()
