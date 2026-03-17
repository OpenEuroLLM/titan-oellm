# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Central import module for all titan_oellm models.

This module imports all custom models to ensure their train specs are registered
and available for training, regardless of which specific model is being used.
"""

# Import all custom models to register their train specs
from . import qwen3_custom
from . import gpt_plus

__all__ = [
    "qwen3_custom",
    "gpt_plus",
]