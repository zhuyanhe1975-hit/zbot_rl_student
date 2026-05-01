# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrapper.

Students should start from zbot_direct/cfg.py. This module remains so older
registry strings or scripts that import task_cfgs keep working.
"""

from ....cfg import *  # noqa: F401,F403
