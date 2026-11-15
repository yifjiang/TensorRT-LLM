# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission boundaries for the ragged target-only watermark extension."""

import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from tensorrt_llm._model_config import StrictBaseModel
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, TorchLlmArgs
from tensorrt_llm.llmapi.utils import StrictBaseModel as LegacyStrictBaseModel
from tensorrt_llm.watermark import WatermarkConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("no_lib_init", ["0", "1"])
@pytest.mark.parametrize("first_import", ["config", "detector"])
def test_watermark_cold_import(first_import: str, no_lib_init: str) -> None:
    """Public watermark imports must work before the eager LLM API is loaded."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TRT_LLM_NO_LIB_INIT"] = no_lib_init
    env["TRTLLM_NO_USAGE_STATS"] = "1"
    code = textwrap.dedent(f"""\
        import sys

        assert "tensorrt_llm" not in sys.modules
        if {first_import!r} == "config":
            from tensorrt_llm.watermark import WatermarkConfig
        else:
            from tensorrt_llm.watermark import GumbelWatermarkDetector
            from tensorrt_llm.watermark import WatermarkConfig

        assert "tensorrt_llm.llmapi" not in sys.modules
        from tensorrt_llm.watermark import GumbelWatermarkDetector
        from tensorrt_llm.watermark.config import WatermarkConfig as DirectConfig
        from tensorrt_llm.watermark.detector import GumbelWatermarkDetector as DirectDetector

        assert DirectConfig is WatermarkConfig
        assert DirectDetector is GumbelWatermarkDetector
        result = GumbelWatermarkDetector(WatermarkConfig(key=7)).detect([1, 2, 3, 4, 5, 6])
        assert result.num_tokens == 6
        assert 0.0 <= result.p_value <= 1.0

        import torch

        assert not torch.cuda.is_initialized()
        print("WATERMARK-COLD-IMPORT-OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WATERMARK-COLD-IMPORT-OK" in result.stdout


def test_watermark_preserves_strict_base_identity_and_serialization() -> None:
    """The legacy import and old pickle globals resolve to the shared base."""
    assert LegacyStrictBaseModel is StrictBaseModel
    assert WatermarkConfig.__bases__ == (StrictBaseModel,)
    assert pickle.loads(b"ctensorrt_llm.llmapi.utils\nStrictBaseModel\n.") is StrictBaseModel
    assert pickle.loads(pickle.dumps(StrictBaseModel)) is StrictBaseModel
    config = WatermarkConfig(key=7)
    assert pickle.loads(pickle.dumps(config)) == config


def test_watermark_preserves_validation_and_key_privacy() -> None:
    """The shared base keeps strict fields and secret-key metadata intact."""
    secret = 0x123456789ABCDEF0
    config = WatermarkConfig(key=secret)
    assert "key=" not in repr(config)
    assert str(secret) not in repr(config)
    schema = WatermarkConfig.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["key"]["telemetry"] == {"exclude": True}
    assert WatermarkConfig.model_fields["key"].repr is False
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WatermarkConfig(key=secret, context_wdith=4)
    with pytest.raises(ValidationError):
        WatermarkConfig(key=2**64)


@pytest.mark.parametrize(
    "algorithm,target_only,rejection",
    [
        ("gumbel", False, True),
        ("gumbel", True, False),
        ("gumbel_dual_key", True, True),
        ("gumbel_dual_key", False, False),
    ],
)
def test_dspark_rejects_unsupported_watermark_modes(algorithm, target_only, rejection):
    """Reject unsafe modes before an executor or GPU is created."""
    args = TorchLlmArgs.model_construct(
        watermark_config=WatermarkConfig(
            key=7, algorithm=algorithm, allow_target_only_watermarking=target_only
        ),
        speculative_config=DSparkDecodingConfig(max_draft_len=7, use_rejection_sampling=rejection),
    )
    with pytest.raises(ValueError, match="DSpark watermarking requires"):
        args.validate_watermark_config()


def test_dspark_accepts_explicit_target_only_rejection_mode():
    """The supported mode preserves ordinary draft and native rejection draws."""
    args = TorchLlmArgs.model_construct(
        watermark_config=WatermarkConfig(key=7, allow_target_only_watermarking=True),
        speculative_config=DSparkDecodingConfig(max_draft_len=7, use_rejection_sampling=True),
    )
    assert args.validate_watermark_config() is args


def test_plain_dspark_does_not_acquire_watermark_restrictions():
    """An engine without watermark configuration retains its prior admission."""
    args = TorchLlmArgs.model_construct(
        watermark_config=None,
        speculative_config=DSparkDecodingConfig(max_draft_len=7, use_rejection_sampling=False),
    )
    assert args.validate_watermark_config() is args
