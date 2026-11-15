# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, process-local watermark construction receipts without tensor values."""

import hashlib
import itertools
import json
import os
import re
import stat
import sys
import time
import uuid
from pathlib import Path

_RECEIPT_ENV = "M31_WM_QUALIFICATION_RECEIPT_DIR"
_STAGE_ORDINAL = itertools.count(1)
_PUBLIC_FIELDS = (
    "algorithm", "prf", "context_width", "deduplicate_contexts",
    "deduplicate_contexts_max_history", "allow_target_only_watermarking", "allow_request_opt_out",
)
_CAPACITY_FIELDS = (
    "max_seq_len", "max_draft_len", "max_num_sequences", "max_beam_width",
    "max_total_draft_tokens", "max_num_requests", "num_seq_slots", "is_cuda_graph",
)
_TENSOR_FIELDS = ("_tail", "_count", "_ring", "_skip", "_keys", "enabled", "drafts", "draft_seeds")


def watermark_key_digest(key: int) -> str:
    """Return the domain-separated identity of an already configured uint64 key."""
    if type(key) is not int or not 0 <= key < 2**64:
        raise ValueError("Watermark identity requires a uint64 key")
    return hashlib.sha256(b"TensorRT-LLM/watermark-key-identity/v1\0" + key.to_bytes(8, "big")).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_origin(name: str) -> dict:
    module = sys.modules.get(name)
    if module is None:
        return {"module": name, "loaded": False}
    filename = getattr(module, "__file__", None)
    if filename is None:
        return {"module": name, "loaded": True, "file": None}
    path = Path(filename).resolve(strict=True)
    return {"module": name, "loaded": True, "file": str(path), "sha256": _sha(path)}


def _class_origin(value: object | None) -> dict | None:
    if value is None:
        return None
    cls = type(value)
    return {"class": cls.__qualname__, "origin": _module_origin(cls.__module__)}


def _fields(value: object | None) -> dict:
    if value is None:
        return {}
    result = {}
    for name in _CAPACITY_FIELDS:
        field = getattr(value, name, None)
        if field is not None:
            if type(field) not in (int, bool):
                raise TypeError("Unexpected construction capacity metadata type")
            result[name] = field
    return result


def _tensor_metadata(value: object | None) -> dict:
    if value is None:
        return {}
    result = {}
    for name in _TENSOR_FIELDS:
        tensor = getattr(value, name, None)
        if tensor is not None:
            result[name] = {"shape": [int(dim) for dim in tensor.shape],
                            "dtype": str(tensor.dtype), "device": str(tensor.device)}
    return result


def _mpi_identity() -> dict:
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is None or not mpi.Is_initialized() or mpi.Is_finalized():
        raise RuntimeError("Requested watermark receipts require an initialized MPI runtime")
    rank, world = mpi.COMM_WORLD.Get_rank(), mpi.COMM_WORLD.Get_size()
    if type(rank) is not int or type(world) is not int or not 0 <= rank < world:
        raise RuntimeError("Invalid actual MPI rank identity")
    ompi = {}
    for name in ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if name in os.environ:
            try:
                ompi[name] = int(os.environ[name])
            except ValueError:
                raise RuntimeError("Invalid OMPI rank environment") from None
    if world > 1 and not {"OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"} <= ompi.keys():
        raise RuntimeError("Distributed watermark receipts require OMPI rank crosschecks")
    if ("OMPI_COMM_WORLD_RANK" in ompi and ompi["OMPI_COMM_WORLD_RANK"] != rank) or (
        "OMPI_COMM_WORLD_SIZE" in ompi and ompi["OMPI_COMM_WORLD_SIZE"] != world
    ):
        raise RuntimeError("Actual MPI rank and OMPI environment disagree")
    return {"rank": rank, "world_size": world, "ompi": ompi}


def _directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Watermark receipt directory must be an absolute owned path")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Watermark receipt directory must not traverse symlinks")
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise PermissionError("Watermark receipt directory must be owned and mode 0700")
        if stat.S_IMODE(info.st_mode) == stat.S_ISGID | 0o700:
            os.fchmod(fd, 0o700)
            info = os.fstat(fd)
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError("Watermark receipt directory must be owned and mode 0700")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _atomic_exclusive(directory: int, filename: str, payload: dict) -> None:
    temporary = ".watermark-" + uuid.uuid4().hex + ".tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, filename, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
    finally:
        os.unlink(temporary, dir_fd=directory)


def emit_watermark_runtime_identity(
    stage: str,
    owner: object,
    *,
    watermark_config: object | None = None,
    sampler_args: object | None = None,
    handler: object | None = None,
    metadata: object | None = None,
    capacity_phase: str = "unclassified",
) -> Path | None:
    """Record one actual construction stage when explicitly requested.

    This observes only the calling process. It does not prove token decisions,
    classify a temporary allocation as final capacity, or inspect tensor contents.
    """
    destination = os.environ.get(_RECEIPT_ENV)
    if not destination:
        return None
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", stage):
        raise ValueError("Invalid watermark construction stage")
    if capacity_phase not in ("unclassified", "profiling", "cuda_graph"):
        raise ValueError("Watermark construction capacity phase must be explicit")
    path = Path(destination)
    directory = _directory(path)
    try:
        mpi = _mpi_identity()
        public, key_identity = None, None
        if watermark_config is not None:
            public = {name: getattr(watermark_config, name) for name in _PUBLIC_FIELDS}
            if any(type(value) not in (str, int, bool, type(None)) for value in public.values()):
                raise TypeError("Unexpected public watermark configuration type")
            key_identity = watermark_key_digest(watermark_config.key)
        ordinal = next(_STAGE_ORDINAL)
        process_stat = Path("/proc/self/stat").read_text()
        start_ticks = int(process_stat.rsplit(")", 1)[1].split()[19])
        owner_id = hex(id(owner))
        payload = {
            "schema_version": 1, "stage": stage, "stage_ordinal": ordinal,
            "stage_id": uuid.uuid4().hex, "observed_ns": time.time_ns(),
            "pid": os.getpid(), "process_start_ticks": start_ticks, "mpi": mpi,
            "owner_instance_id": owner_id, "capacity_phase": capacity_phase,
            "final_capacity_claim": False, "watermark_enabled": watermark_config is not None,
            "watermark_public_config": public, "watermark_key_digest": key_identity,
            "public_config_fingerprint": None if public is None else hashlib.sha256(
                json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "classes": {"owner": _class_origin(owner), "config": _class_origin(watermark_config),
                        "sampler_args": _class_origin(sampler_args), "handler": _class_origin(handler),
                        "metadata": _class_origin(metadata)},
            "capacities": {"owner": _fields(owner), "sampler_args": _fields(sampler_args),
                           "metadata": _fields(metadata)},
            "handler_tensor_metadata": _tensor_metadata(handler),
            "metadata_tensor_metadata": _tensor_metadata(metadata),
            "loaded_feature_modules": [_module_origin(name) for name in (
                "tensorrt_llm.bindings", "tensorrt_llm._torch.pyexecutor.sampler.watermark",
                "tensorrt_llm._torch.pyexecutor.sampler.watermark_runtime",
                "tensorrt_llm.watermark.config", "tensorrt_llm.watermark.prf",
                "tensorrt_llm.watermark.sampling", "tensorrt_llm.watermark.state_kernels",
                "tensorrt_llm.watermark.step_kernel",
            )],
            "scope": "Actual in-process construction identity; no token-step or final-capacity proof",
        }
        filename = f"{stage}-rank{mpi['rank']}-pid{os.getpid()}-stage{ordinal}.json"
        _atomic_exclusive(directory, filename, payload)
        return path / filename
    finally:
        os.close(directory)
