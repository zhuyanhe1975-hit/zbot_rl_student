# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal task implementation package.

Gym registration lives at zbot_direct/source/zbot_direct/zbot_direct/__init__.py
so students can start from the top-level env.py and cfg.py files. This package
keeps the implementation modules and remains importable for old paths.
"""

from . import agents

from .... import register_tasks

register_tasks()

__all__ = ["agents"]
