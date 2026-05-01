# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student-facing velocity task settings.

These are the values students most often want to inspect or tune before going
into the lower-level env implementation.
"""

ZBOT_VELOCITY_COMMAND_RANGES = {
    "lin_vel_x": (-0.8, 0.8),
    "lin_vel_y": (-0.4, 0.4),
    "ang_vel_z": (-1.0, 1.0),
}

ZBOT_VELOCITY_CURRICULUM = {
    "standing_probability": 0.2,
    "command_resample_time_range": (4.0, 8.0),
    "stepping_frequency_range": (0.8, 1.6),
    "stepping_clearance_target": 0.10,
    "curriculum_warmup_steps": 2000,
    "curriculum_transition_steps": 8000,
    "yaw_tracking_start_ratio": 0.45,
    "xy_command_min_curriculum_scale": 0.25,
    "xy_command_curriculum_power": 1.5,
}
