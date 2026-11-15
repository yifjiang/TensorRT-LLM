# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep request-level watermark opt-out intact across API and rank boundaries."""

import json
import pickle
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
from tensorrt_llm._torch.pyexecutor.request_utils import (
    RequestBroadcaster,
    attach_py_objects_to_requests,
)
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest

pytestmark = pytest.mark.cpu_only


def _api_request(kind: str, **extra) -> CompletionRequest | ChatCompletionRequest:
    if kind == "completion":
        return CompletionRequest(model="test-model", prompt="hello", max_tokens=4, **extra)
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=4,
        **extra,
    )


@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("watermarking", [None, True, False])
def test_openai_watermarking_round_trip_to_sampling_params(
    kind: str, watermarking: bool | None
) -> None:
    request = _api_request(kind, watermarking=watermarking)
    payload = request.model_dump_json(exclude_unset=True)
    assert json.loads(payload)["watermarking"] is watermarking
    restored = type(request).model_validate_json(payload)

    assert restored.watermarking is watermarking
    assert restored.to_sampling_params(backend="pytorch").watermarking is watermarking


@pytest.mark.parametrize("kind", ["completion", "chat"])
def test_omitted_openai_watermarking_preserves_engine_default(kind: str) -> None:
    request = _api_request(kind)
    payload = request.model_dump_json(exclude_unset=True)
    assert "watermarking" not in json.loads(payload)
    restored = type(request).model_validate_json(payload)

    assert restored.watermarking is None
    assert restored.to_sampling_params(backend="pytorch").watermarking is None


@pytest.mark.parametrize("watermarking", [None, True, False])
def test_generation_request_pickle_preserves_watermarking(watermarking: bool | None) -> None:
    request = GenerationRequest(
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4, watermarking=watermarking),
    )

    restored = pickle.loads(pickle.dumps(request))

    assert restored.prompt_token_ids == [1, 2, 3]
    assert restored.sampling_params.watermarking is watermarking


@pytest.mark.parametrize("watermarking", [None, True, False])
def test_base_worker_attaches_watermarking_to_native_request(watermarking: bool | None) -> None:
    captured = []
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = True
    worker._lora_manager = None
    worker._is_pytorch_backend = True
    worker.engine = SimpleNamespace(
        mapping=SimpleNamespace(cp_size=1),
        max_seq_len=32,
        enqueue_request=lambda request: captured.append(request) or 7,
    )
    request = GenerationRequest(
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4, watermarking=watermarking),
    )
    request.id = 7

    assert worker._enqueue_request(request) == 7
    assert len(captured) == 1
    assert isinstance(captured[0], trtllm.Request)
    assert captured[0].py_watermarking is watermarking


def test_watermarking_survives_native_request_and_cross_rank_metadata_pickle() -> None:
    expected = [None, True, False, None]
    source = []
    for index, value in enumerate(expected):
        request = trtllm.Request(input_token_ids=[1, 2, 3], max_tokens=4)
        request.py_conversation_params = None
        if index != 3:
            request.py_watermarking = value
        source.append(RequestQueueItem(index + 1, request))
    metadata = RequestBroadcaster._collect_py_objects(None, source)
    assert dict(metadata)["py_watermarking"] == dict(enumerate(expected, start=1))

    target = [
        RequestQueueItem(item.id, pickle.loads(pickle.dumps(item.request))) for item in source
    ]
    attach_py_objects_to_requests(target, pickle.loads(pickle.dumps(metadata)))

    for item, value in zip(target, expected):
        assert item.request.py_watermarking is value
        converted = executor_request_to_llm_request(
            item.id, item.request, child_req_ids=[], exclude_last_generation_logits=False
        )
        assert converted.py_watermarking is value
        assert converted.py_watermark_neutral is None


@pytest.mark.parametrize("watermarking", [None, True, False])
def test_child_request_preserves_watermarking_and_has_its_own_cache(
    watermarking: bool | None,
) -> None:
    native = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=4,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
    )
    native.py_watermarking = watermarking
    request = executor_request_to_llm_request(
        42, native, child_req_ids=[], exclude_last_generation_logits=False
    )
    request.py_watermark_neutral = False

    request.create_child_request(43)

    assert request.py_watermarking is watermarking
    assert request.py_watermark_neutral is False
    assert request.child_requests[0].py_watermarking is watermarking
    assert request.child_requests[0].py_watermark_neutral is None
