# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 mixed-request and streaming watermark coverage with stage-matched controls.

Each engine runs in a fresh subprocess. The reference repeats the same prompts
at the same generation stages: a warm KV-cache hit need not reproduce the cold
request's bf16 logits. Set TRTLLM_WATERMARK_TP4_MODEL to override the model and
TRTLLM_WATERMARK_TP4_OUTPUT_DIR to retain all raw outputs and detector scores.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from tensorrt_llm import SamplingParams
    from tensorrt_llm.llmapi.llm import RequestOutput

_MODES = ("baseline", "gumbel", "gumbel_dual_key")
_KEY = 123456789
_P_VALUE_THRESHOLD = 1e-3
_MAX_TOKENS = 256
_PROMPTS = [
    "Write an imaginative travel journal about a botanist arriving at a floating city. "
    "Include concrete sensory details.",
    "Explain how a bicycle works to an interested teenager, with practical details about energy "
    "and gears.",
    "Tell an original story about a mapmaker who discovers an island that changes shape every "
    "morning.",
    "Describe a possible future scientific expedition to an ocean beneath an icy moon, with "
    "vivid practical details.",
]


def _save(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _params(index: int, **updates: object) -> SamplingParams:
    from tensorrt_llm import SamplingParams

    values = dict(
        seed=431 + index,
        temperature=0.9,
        top_p=0.98,
        top_k=100,
        min_p=0.01,
        max_tokens=_MAX_TOKENS,
        ignore_eos=True,
        watermarking=False,
    )
    values.update(updates)
    return SamplingParams(**values)


def _pack(output: RequestOutput) -> dict:
    return {
        "prompt_token_ids": list(output.prompt_token_ids),
        "token_ids": list(output.outputs[0].token_ids),
        "text": output.outputs[0].text,
    }


def _sampling_record(params: SamplingParams, index: int) -> dict:
    names = (
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "ignore_eos",
        "watermarking",
    )
    return {"prompt_index": index, "parameters": {name: getattr(params, name) for name in names}}


def _worker(mode: str, model: str, destination: Path) -> None:
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.watermark import WatermarkConfig

    destination.mkdir(parents=True)
    path = destination / "RESULT.json"
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in _PROMPTS
    ]
    assert all(
        isinstance(prompt, list) and all(isinstance(token, int) for token in prompt)
        for prompt in prompts
    )
    config = None if mode == "baseline" else WatermarkConfig(algorithm=mode, key=_KEY)
    result = {
        "mode": mode,
        "model": model,
        "tp_size": 4,
        "status": "IN_PROGRESS",
        "prompt_token_ids": prompts,
        "sampling_parameters": {},
        "overlap_scheduler_enabled": True,
        "cuda_graphs_enabled": True,
        "in_graph_sampling_enabled": False,
        "block_reuse_enabled": True,
    }
    _save(path, result)
    started = time.monotonic()
    llm = None
    try:
        llm = LLM(
            model=model,
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_batch_size=8,
            max_seq_len=1024,
            max_num_tokens=2048,
            kv_cache_config=KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.2),
            cuda_graph_config={"batch_sizes": [1, 2, 4, 8]},
            disable_overlap_scheduler=False,
            enable_in_graph_sampling=False,
            generation_config="trtllm",
            watermark_config=config,
        )
        for stage in ("plain", "plain_repeat", "greedy", "marked", "mixed"):
            indices = range(2) if stage == "greedy" else range(4)
            parameters = []
            for index in indices:
                updates = {}
                if stage == "greedy":
                    updates = {"temperature": 0.0, "max_tokens": 64, "watermarking": None}
                elif config is not None and stage == "marked":
                    updates = {"watermarking": None}
                elif config is not None and stage == "mixed":
                    updates = {"watermarking": index % 2 == 0}
                parameters.append(_params(index, **updates))
            result["sampling_parameters"][stage] = [
                _sampling_record(params, index) for index, params in zip(indices, parameters)
            ]
            result[stage] = [
                _pack(output)
                for output in llm.generate(
                    [prompts[index] for index in indices], parameters, use_tqdm=False
                )
            ]
            result["last_completed_stage"] = stage
            _save(path, result)
            print(f"WATERMARK_TP4_STAGE mode={mode} stage={stage}", flush=True)

        async def stream() -> None:
            params = _params(17, watermarking=True if config is not None else False)
            result["sampling_parameters"]["streaming"] = [_sampling_record(params, 0)]
            result["streaming"] = {"chunks": 0, "output": None}
            async for output in llm.generate_async(prompts[0], params, streaming=True):
                result["streaming"]["chunks"] += 1
                result["streaming"]["output"] = _pack(output)

        asyncio.run(stream())
        result["last_completed_stage"] = "streaming"
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        _save(path, result)
        if llm is not None:
            llm.shutdown()


def _detections(config: object, rows: list[dict]) -> list[dict]:
    from tensorrt_llm.watermark import GumbelWatermarkDetector

    detector = GumbelWatermarkDetector(config, p_value_threshold=_P_VALUE_THRESHOLD)
    return [
        dataclasses.asdict(
            detector.detect(row["token_ids"], prompt_token_ids=row["prompt_token_ids"])
        )
        for row in rows
    ]


def _rank_agreement(trace: Path, mode: str) -> dict:
    identities = [
        json.loads((trace / f"identity-rank-{rank}.json").read_text()) for rank in range(4)
    ]
    records = [
        [json.loads(line) for line in (trace / f"rank-{rank}.jsonl").read_text().splitlines()]
        for rank in range(4)
    ]
    assert all(
        identity["rank"] == rank and identity["world_size"] == 4
        for rank, identity in enumerate(identities)
    )
    assert len(records[0]) >= _MAX_TOKENS
    assert all(record == records[0] for record in records[1:])
    assert all(record["algorithm"] == mode for record in records[0])
    return {
        "ranks": 4,
        "sampling_calls": len(records[0]),
        "tokens_compared_per_rank": sum(len(record["tokens"]) for record in records[0]),
    }


@pytest.mark.gpu4
@pytest.mark.high_cuda_memory
def test_watermark_tp4_mixed_opt_out_and_streaming(tmp_path: Path) -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip("TP4 requires four CUDA devices")
    from utils.llm_data import llm_models_root

    from tensorrt_llm.watermark import WatermarkConfig

    model = os.environ.get("TRTLLM_WATERMARK_TP4_MODEL")
    if model is None:
        model_root = llm_models_root()
        if model_root is None:
            pytest.skip("LLM_MODELS_ROOT is not set")
        model = str(Path(model_root) / "Qwen3/Qwen3-8B")
    if not Path(model).is_dir():
        pytest.skip(f"Watermark TP4 model is unavailable: {model}")
    output = Path(os.environ.get("TRTLLM_WATERMARK_TP4_OUTPUT_DIR", str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    assert not any((output / mode).exists() for mode in _MODES), "Use a fresh output directory"
    trace_root = os.environ.get("TRTLLM_WATERMARK_TP4_TRACE_ROOT")
    for mode in _MODES:
        env = dict(os.environ)
        env.pop("WM_TRACE_DIR", None)
        if trace_root is not None and mode != "baseline":
            env["WM_TRACE_DIR"] = str(Path(trace_root) / f"traces-{mode}")
        with (output / f"{mode}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--mode",
                    mode,
                    "--model",
                    model,
                    "--output",
                    str(output / mode),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )

    baseline = json.loads((output / "baseline/RESULT.json").read_text())
    assessment = {"modes": {}, "checks": {}}
    for mode in _MODES[1:]:
        raw = json.loads((output / mode / "RESULT.json").read_text())
        checks = {}
        for stage in ("plain", "plain_repeat", "greedy"):
            checks[f"{stage}_matches_reference"] = raw[stage] == baseline[stage]
        checks["mixed_optout_matches_warm_reference"] = all(
            raw["mixed"][index] == baseline["mixed"][index] for index in (1, 3)
        )
        checks["streamed_final_tokens_complete"] = (
            raw["streaming"]["chunks"] > 1
            and len(raw["streaming"]["output"]["token_ids"]) == _MAX_TOKENS
        )
        config = WatermarkConfig(algorithm=mode, key=_KEY)
        wrong_config = WatermarkConfig(algorithm=mode, key=_KEY + 1)
        scores = {}
        for name, rows in (
            ("marked", raw["marked"]),
            ("mixed_marked", [raw["mixed"][index] for index in (0, 2)]),
            ("streaming", [raw["streaming"]["output"]]),
        ):
            scores[name] = _detections(config, rows)
            scores[f"{name}_wrong_key"] = _detections(wrong_config, rows)
            checks[f"{name}_detected"] = sum(
                item["is_watermarked"] for item in scores[name]
            ) * 2 >= len(rows)
            checks[f"{name}_wrong_key_control"] = sum(
                item["is_watermarked"] for item in scores[f"{name}_wrong_key"]
            ) <= (1 if len(rows) > 1 else 0)
        for name, rows in (
            ("plain", raw["plain"]),
            ("mixed_optout", [raw["mixed"][index] for index in (1, 3)]),
        ):
            scores[name] = _detections(config, rows)
            checks[f"{name}_unmarked_control"] = (
                sum(item["is_watermarked"] for item in scores[name]) <= 1
            )
        scores["greedy"] = _detections(config, raw["greedy"])
        if trace_root is not None:
            try:
                scores["rank_agreement"] = _rank_agreement(
                    Path(trace_root) / f"traces-{mode}", mode
                )
            except (AssertionError, OSError, ValueError, KeyError, TypeError) as exc:
                scores["rank_agreement"] = {"error": repr(exc)}
                checks["rank_agreement"] = False
            else:
                checks["rank_agreement"] = True
        assessment["modes"][mode] = scores
        assessment["checks"][mode] = checks
    assessment["status"] = (
        "PASS" if all(all(checks.values()) for checks in assessment["checks"].values()) else "FAIL"
    )
    _save(output / "assessment.json", assessment)
    assert assessment["status"] == "PASS", (
        f"Watermark TP4 checks failed; see {output / 'assessment.json'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _worker(arguments.mode, arguments.model, arguments.output)
