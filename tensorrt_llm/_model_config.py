# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration model base without LLM API dependencies."""

from pydantic import BaseModel


class StrictBaseModel(BaseModel):
    """
    A base model that forbids arbitrary fields.

    All user-facing configuration classes should inherit from this to ensure
    typos and invalid fields are caught at validation time.
    """

    class Config:
        extra = "forbid"
