# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student-facing reward scale tables.

Edit these complete dictionaries directly. Task cfg classes import the same
objects, so changes here are used by training without a separate sync step.
"""

ZBOT_6DOF_VELOCITY_REWARD_SCALES = {
    "track_lin_vel_xy_exp": 4.0,
    "track_ang_vel_z_exp": 1.0,
    "yaw_rate_l1": -0.5,
    "command_forward_progress": 2.0,
    "command_speed_shortfall": -1.0,
    "lateral_velocity_match": 0.5,
    "yaw_when_translating": -0.4,
    "command_stillness": -1.0,
    "ang_vel_xy_l2": -0.05,
    "leg_toggle": 5.0,
    "single_leg_support": 0.8,
    "airtime_balance": -1.0,
    "double_support": -0.4,
    "action_rate_l2": -0.02,
    "dof_torques_l2": -2.0e-4,
    "joint_deviation_l1": -0.05,
    "feet_slide": -0.5,
}
DEFAULT_VELOCITY_REWARD_SCALES = ZBOT_6DOF_VELOCITY_REWARD_SCALES

ZBOT_8DOF_VELOCITY_REWARD_SCALES = {
    "track_lin_vel_xy_exp": 8.0,
    "track_ang_vel_z_exp": 0.5,
    "command_forward_progress": 2.0,
    "command_speed_shortfall": -1.0,
    "lateral_velocity_match": 0.5,
    "command_stillness": -0.2,
    "ang_vel_xy_l2": -0.05,
    "leg_toggle": 5.0,
    "single_leg_support": 0.8,
    "airtime_balance": -1.0,
    "double_support": -0.4,
    "action_rate_l2": -0.02,
    "dof_torques_l2": -2.0e-4,
    "joint_deviation_l1": -0.05,
    "feet_slide": -0.5,
    "feet_forward_bias_integral": -0.05,
    "feet_downward": -1.0,
    "feet_forward": -0.5,
}

ZBOT_6DOF_VELOCITY_NON_DEFAULT_REWARDS = {}
ZBOT_8DOF_VELOCITY_NON_DEFAULT_REWARDS = {
    name: scale
    for name, scale in ZBOT_8DOF_VELOCITY_REWARD_SCALES.items()
    if DEFAULT_VELOCITY_REWARD_SCALES.get(name) != scale
}
