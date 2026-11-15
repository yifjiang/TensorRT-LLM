# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Privacy, process identity and file-admission boundaries of optional receipts."""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.sampler import watermark_runtime as runtime
from tensorrt_llm.watermark import WatermarkConfig

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def receipt_dir(tmp_path, monkeypatch):
    path = tmp_path.resolve() / "receipts"
    monkeypatch.setenv("M31_WM_QUALIFICATION_RECEIPT_DIR", str(path))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "2")
    comm = SimpleNamespace(Get_rank=lambda: 2, Get_size=lambda: 4)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", SimpleNamespace(
        Is_initialized=lambda: True, Is_finalized=lambda: False, COMM_WORLD=comm
    ))
    return path


def test_unset_receipt_env_does_not_inspect_objects_or_files(monkeypatch):
    class Untouchable:
        def __getattribute__(self, name):
            raise AssertionError("Disabled receipts must not inspect arguments")

    monkeypatch.delenv("M31_WM_QUALIFICATION_RECEIPT_DIR", raising=False)
    monkeypatch.setattr(runtime, "_directory", lambda path: pytest.fail("Unexpected file access"))
    assert runtime.emit_watermark_runtime_identity("invalid stage", Untouchable(),
                                                    watermark_config=Untouchable()) is None


def test_actual_public_config_key_privacy_and_self_process_identity(receipt_dir):
    key = 2**63 + 12345
    config = WatermarkConfig(key=key, allow_target_only_watermarking=True)
    owner = SimpleNamespace(max_num_sequences=4, unrelated_secret="never serialize this")
    path = runtime.emit_watermark_runtime_identity(
        "torch_sampler_constructed", owner, watermark_config=config, sampler_args=owner
    )
    raw = path.read_text()
    receipt = json.loads(raw)
    assert str(key) not in raw and "never serialize this" not in raw
    assert receipt["watermark_public_config"] == config.model_dump(exclude={"key"})
    expected_digest = hashlib.sha256(
        b"TensorRT-LLM/watermark-key-identity/v1\0" + key.to_bytes(8, "big")
    ).hexdigest()
    assert receipt["watermark_key_digest"] == expected_digest
    assert receipt["pid"] == os.getpid() and receipt["process_start_ticks"] > 0
    assert receipt["mpi"]["rank"] == 2 and receipt["mpi"]["world_size"] == 4
    origin = receipt["classes"]["config"]["origin"]
    assert origin["loaded"] and origin["sha256"] == hashlib.sha256(Path(origin["file"]).read_bytes()).hexdigest()
    assert stat.S_IMODE(receipt_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_engine_off_has_no_fabricated_config_or_key(receipt_dir):
    receipt = json.loads(runtime.emit_watermark_runtime_identity("spec_sampler_constructed", object()).read_text())
    assert receipt["watermark_enabled"] is False
    assert receipt["watermark_public_config"] is None
    assert receipt["watermark_key_digest"] is None
    assert receipt["public_config_fingerprint"] is None


def test_tensor_metadata_never_materializes_device_values(receipt_dir):
    class TensorMetadata:
        shape = (5, 4)
        dtype = "torch.int64"
        device = "cuda:2"

        def tolist(self):
            raise AssertionError("Tensor contents must not be inspected")

        item = cpu = numpy = tolist

    handler = SimpleNamespace(_tail=TensorMetadata(), secret_value=123)
    path = runtime.emit_watermark_runtime_identity(
        "spec_metadata_prepared", handler, handler=handler, capacity_phase="cuda_graph"
    )
    receipt = json.loads(path.read_text())
    assert receipt["handler_tensor_metadata"] == {
        "_tail": {"shape": [5, 4], "dtype": "torch.int64", "device": "cuda:2"}
    }
    assert receipt["capacity_phase"] == "cuda_graph" and receipt["final_capacity_claim"] is False


def test_repeated_stages_keep_instance_identity_and_advance_ordinal(receipt_dir):
    owner = object()
    paths = [runtime.emit_watermark_runtime_identity("torch_sampler_constructed", owner) for _ in range(2)]
    first, second = [json.loads(path.read_text()) for path in paths]
    assert paths[0] != paths[1] and first["owner_instance_id"] == second["owner_instance_id"]
    assert first["stage_ordinal"] < second["stage_ordinal"] and first["stage_id"] != second["stage_id"]
    assert first["capacity_phase"] == second["capacity_phase"] == "unclassified"
    assert not first["final_capacity_claim"] and not second["final_capacity_claim"]


@pytest.mark.parametrize("failure", ["relative", "parent_traversal", "symlink", "insecure", "file"])
def test_requested_invalid_directory_fails_closed(receipt_dir, monkeypatch, failure):
    if failure == "relative":
        monkeypatch.setenv("M31_WM_QUALIFICATION_RECEIPT_DIR", "relative")
    elif failure == "parent_traversal":
        monkeypatch.setenv("M31_WM_QUALIFICATION_RECEIPT_DIR", str(receipt_dir.parent / ".." / "receipts"))
    elif failure == "symlink":
        target = receipt_dir.with_name("actual")
        target.mkdir(mode=0o700)
        receipt_dir.symlink_to(target, target_is_directory=True)
    elif failure == "insecure":
        receipt_dir.mkdir(mode=0o755)
        receipt_dir.chmod(0o755)
    else:
        receipt_dir.write_text("not a directory")
    with pytest.raises((ValueError, PermissionError, OSError)):
        runtime.emit_watermark_runtime_identity("torch_sampler_constructed", object())


def test_requested_directory_owned_by_another_uid_is_rejected(receipt_dir, monkeypatch):
    receipt_dir.mkdir(mode=0o700)
    receipt_dir.chmod(0o2700)
    original_mode = stat.S_IMODE(receipt_dir.stat().st_mode)
    uid = os.getuid()
    monkeypatch.setattr(runtime.os, "getuid", lambda: uid + 1)
    with pytest.raises(PermissionError, match="owned and mode 0700"):
        runtime.emit_watermark_runtime_identity("torch_sampler_constructed", object())
    assert stat.S_IMODE(receipt_dir.stat().st_mode) == original_mode


def test_owned_owner_only_directory_clears_inherited_sgid(receipt_dir):
    receipt_dir.mkdir(mode=0o700)
    receipt_dir.chmod(0o2700)
    fd = runtime._directory(receipt_dir)
    try:
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o700
        assert os.fstat(fd).st_uid == os.getuid()
    finally:
        os.close(fd)


@pytest.mark.parametrize("mode", [0o2750, 0o3700, 0o4700, 0o2600])
def test_special_or_shared_directory_modes_are_never_relaxed(receipt_dir, mode):
    receipt_dir.mkdir(mode=0o700)
    receipt_dir.chmod(mode)
    original_mode = stat.S_IMODE(receipt_dir.stat().st_mode)
    with pytest.raises(PermissionError, match="owned and mode 0700"):
        runtime.emit_watermark_runtime_identity("torch_sampler_constructed", object())
    assert stat.S_IMODE(receipt_dir.stat().st_mode) == original_mode


@pytest.mark.parametrize("failure", ["rank_disagrees", "size_missing", "mpi_missing", "mpi_uninitialized"])
def test_actual_mpi_and_ompi_crosscheck_fails_closed(receipt_dir, monkeypatch, failure):
    if failure == "rank_disagrees":
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    elif failure == "size_missing":
        monkeypatch.delenv("OMPI_COMM_WORLD_SIZE")
    elif failure == "mpi_missing":
        monkeypatch.delitem(sys.modules, "mpi4py.MPI")
    else:
        monkeypatch.setattr(sys.modules["mpi4py.MPI"], "Is_initialized", lambda: False)
    with pytest.raises(RuntimeError):
        runtime.emit_watermark_runtime_identity("torch_sampler_constructed", object())
    assert not list(receipt_dir.glob("*.json"))


def test_atomic_exclusive_receipt_never_replaces_an_existing_file(receipt_dir):
    receipt_dir.mkdir(mode=0o700)
    fd = runtime._directory(receipt_dir)
    try:
        runtime._atomic_exclusive(fd, "fixed.json", {"first": True})
        original = (receipt_dir / "fixed.json").read_bytes()
        with pytest.raises(FileExistsError):
            runtime._atomic_exclusive(fd, "fixed.json", {"replacement": True})
        assert (receipt_dir / "fixed.json").read_bytes() == original
        assert sorted(path.name for path in receipt_dir.iterdir()) == ["fixed.json"]
    finally:
        os.close(fd)


@pytest.mark.parametrize("key", [-1, 2**64, True])
def test_key_identity_does_not_coerce_invalid_values(key):
    with pytest.raises(ValueError, match="uint64"):
        runtime.watermark_key_digest(key)
