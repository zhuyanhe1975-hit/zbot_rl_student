# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student-facing task configuration file.

Start here to compare tasks. Each task cfg keeps the important differences in
one place: robot asset, action/observation size, reward scales, command ranges,
and termination settings. Shared robot USD definitions live in
tasks/direct/zbot_direct/zbot_direct_robot_cfgs.py.

Quick guide:
- Change reward weights here.
- For velocity tasks, edit the full *_REWARD_SCALES dictionaries below; cfg classes use them directly.
- Change robot assets here, then inspect zbot_direct_robot_cfgs.py if needed.
- Add a new task cfg here, then register it in zbot_direct/source/zbot_direct/zbot_direct/__init__.py.
"""

import math

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .reward_scales import (
    ZBOT_6DOF_VELOCITY_REWARD_SCALES,
    ZBOT_8DOF_VELOCITY_REWARD_SCALES,
)
from .tasks.direct.zbot_direct.velocity_env_cfg import ZbotVelocityEnvCfg
from .tasks.direct.zbot_direct.zbot_direct_robot_cfgs import (
    ZBOT_D_6S_CFG,
    ZBOT_D_8S_HUMAN_2_BIPEDAL_2_CFG,
    ZBOT_D_8S_HUMAN_2_BIPEDAL_3_CFG,
    ZBOT_D_8S_HUMAN_BIPEDAL_1_CFG,
    ZBOT_D_8S_HUMAN_CFG,
    ZBOT_D_8S_HUMAN_NEUTRAL_CFG,
    ZBOT_D_8S_NEW_NEUTRAL_CFG,
    ZBOT_D_8S_SNAKE_V0_CFG,
    ZBOT_D_8S_WHEEL_CFG,
)


ROBOT_PRIM_PATH = "/World/envs/env_.*/Robot"


@configclass
class ZbotDirectTaskCfg(DirectRLEnvCfg):
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"{ROBOT_PRIM_PATH}/.*",
        history_length=5,
        update_period=0.0,
        track_air_time=True,
        track_pose=True,
    )

    episode_length_s = 20.0
    decimation = 2
    action_scale = 1.0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )


@configclass
class Zbot6DofTaskCfg(ZbotDirectTaskCfg):
    robot: ArticulationCfg = ZBOT_D_6S_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    action_space = 6
    termination_height = 0.22


@configclass
class Zbot8DofTaskCfg(ZbotDirectTaskCfg):
    action_space = 8
    observation_space = 44
    termination_height = 0.3
    moving_direction = -1.0


@configclass
class Zbot6DofBipedalCfg(Zbot6DofTaskCfg):
    observation_space = 24
    moving_direction = 1.0
    reward_cfg = {
        "reward_scales": {
            "feet_downward": -1.0,
            "feet_forward": -0.5,
            "base_heading_x": -1.0,
            "feet_force_diff": 2.0,
            "feet_force_sum": -0.1,
        },
    }
    curriculum_cfg = {
        "stage_1_reward_scales": {
            "feet_downward": -1.0,
            "feet_forward": -0.5,
            "base_heading_x": -1.0,
            "feet_force_diff": 2.0,
            "feet_force_sum": -0.1,
        },
        "stage_2_reward_scales": {
            "base_vel_forward": 5.0,
            "feet_downward": -1.0,
            "feet_forward": -0.5,
            "similar_to_default": -0.5,
            "base_heading_x": -1.0,
            "base_heading_x_sum": -1.0,
            "step_length": 5.0,
            "airtime_balance": -15.0,
            "airtime_sum": 2.0,
            "action_rate": -0.1,
            "torques": -1.0,
            "feet_slide": -10.0,
            "base_pos_y_err": -0.1,
        },
        "promotion_metric": "feet_force_diff",
        "promotion_threshold": 0.5,
        "promotion_window_steps": 50,
    }


@configclass
class Zbot6DofBipedalQuatCfg(Zbot6DofTaskCfg):
    observation_space = 26
    moving_direction = 1.0
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 1.0,
            "feet_downward": -1.0,
            "feet_forward": -0.5,
        },
    }


@configclass
class Zbot6DofBipedalToSnakeCfg(Zbot6DofTaskCfg):
    observation_space = 44
    moving_direction = 1.0
    reward_cfg = {
        "reward_scales": {
            "feet_downward": -1.0,
            "base_heading_x": -1.0,
            "action_rate": -0.1,
            "torques": -1.0,
            "shape_symmetry": -1.0,
            "base_height": -5.0,
        },
    }


@configclass
class Zbot6DofBipedalToSnake1Cfg(Zbot6DofTaskCfg):
    observation_space = 44
    moving_direction = 1.0
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 10.0,
            "feet_downward": -1.0,
            "base_heading_x": -1.0,
            "action_rate": -0.1,
            "torques": -1.0,
            "shape_symmetry": -1.0,
            "base_height": -5.0,
        },
    }


@configclass
class Zbot8DofBipedal0Cfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_HUMAN_NEUTRAL_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 5.0,
            "similar_to_default": -0.2,
            "base_heading_x": -0.5,
            "base_heading_x_sum": -2.0,
            "airtime_balance": -2.0,
            "torques": -0.2,
        },
    }


@configclass
class Zbot8DofBipedal1Cfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_HUMAN_BIPEDAL_1_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 10.0,
            "similar_to_default": -0.2,
            "base_heading_x": -0.5,
            "base_heading_x_sum": -2.0,
            "airtime_balance": -2.0,
            "torques": -0.2,
            "joint_acc": -0.05,
        },
    }


@configclass
class Zbot8DofBipedal2Cfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_HUMAN_2_BIPEDAL_2_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 10.0,
            "similar_to_default": -0.1,
            "base_heading_x": -0.5,
            "base_heading_x_sum": -2.0,
            "airtime_balance": -2.0,
            "torques": -0.2,
            "joint_acc": -0.05,
        },
    }


@configclass
class Zbot8DofBipedal3Cfg(Zbot8DofBipedal2Cfg):
    robot: ArticulationCfg = ZBOT_D_8S_HUMAN_2_BIPEDAL_3_CFG.replace(prim_path=ROBOT_PRIM_PATH)


@configclass
class Zbot8DofBipedalCfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_HUMAN_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 2.0,
            "feet_forward": -0.1,
            "similar_to_default": -0.2,
            "base_heading_x": -0.5,
            "base_heading_x_sum": -2.0,
            "step_length": 10.0,
            "airtime_balance": -2.0,
            "action_rate": -0.02,
            "torques": -0.2,
            "feet_slide": -2.0,
            "base_pos_y_err": -0.2,
            "base_vel_side": -1.0,
        },
    }


@configclass
class Zbot8DofBirdCfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_NEW_NEUTRAL_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 5.0,
            "base_heading_x": -0.5,
            "base_heading_x_sum": -2.0,
            "step_length": 10.0,
        },
    }


@configclass
class Zbot8DofSnakeCfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_SNAKE_V0_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    termination_height = 0.0
    moving_direction = 1.0
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 5.0,
            "similar_to_default": -0.1,
            "close_ground": 0.5,
            "joint_acc": -0.1,
        },
    }


@configclass
class Zbot8DofWheelCfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_WHEEL_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    termination_height = 0.0
    moving_direction = 1.0
    reward_cfg = {
        "reward_scales": {
            "com_vel_y": 15.0,
            "similar_to_default": -1.0,
            "torques": -0.2,
        },
    }


@configclass
class Zbot6DofVelocityCfg(ZbotVelocityEnvCfg):
    robot: ArticulationCfg = ZBOT_D_6S_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    action_space = 6
    observation_space = 33
    reward_scales = ZBOT_6DOF_VELOCITY_REWARD_SCALES


@configclass
class Zbot6DofVelocityQuatTeacherCfg(Zbot6DofVelocityCfg):
    observation_space = 37
    teacher_observes_base_quat = True


@configclass
class Zbot6DofVelocityImuCfg(Zbot6DofVelocityCfg):
    observation_space = 34
    policy_observes_base_lin_vel = False
    teacher_observes_base_quat = True
    provide_teacher_observations = True


@configclass
class Zbot8DofVelocityCfg(ZbotVelocityEnvCfg):
    robot: ArticulationCfg = ZBOT_D_8S_HUMAN_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    action_space = 8
    observation_space = 39

    reference_joint_pos = (
        -0.08,
        0.0,
        math.radians(25.0),
        math.radians(-125.0),
        math.radians(125.0),
        math.radians(-25.0),
        0.0,
        0.08,
    )
    joint_target_lower = tuple(q - 0.5 * math.pi for q in reference_joint_pos)
    joint_target_upper = tuple(q + 0.5 * math.pi for q in reference_joint_pos)

    reward_scales = ZBOT_8DOF_VELOCITY_REWARD_SCALES
