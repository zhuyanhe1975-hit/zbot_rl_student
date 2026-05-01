# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student-facing task environments.

Start here when you want to understand which environment implementation each
task uses. The reusable mechanics live under
tasks/direct/zbot_direct/: base_env.py, bipedal_env.py, transition_env.py,
ground_env.py, velocity_env.py.

Quick guide:
- To add a task, add a small class here, then add its cfg in cfg.py.
- To inspect rewards, Ctrl+click the class's cfg_cls and read reward_cfg or reward_scales.
- For velocity tasks, Ctrl+click reward_scales_link to see the complete reward weights directly.
- To change observations, follow the parent class named by the task below.
- Velocity play tasks use keyboard commands and should run with num_envs=1.
"""

from __future__ import annotations

from .cfg import (
    Zbot6DofBipedalCfg,
    Zbot6DofBipedalQuatCfg,
    Zbot6DofBipedalToSnake1Cfg,
    Zbot6DofBipedalToSnakeCfg,
    Zbot6DofVelocityCfg,
    Zbot6DofVelocityImuCfg,
    Zbot6DofVelocityQuatTeacherCfg,
    ZBOT_8DOF_VELOCITY_REWARD_SCALES,
    Zbot8DofBipedal0Cfg,
    Zbot8DofBipedal1Cfg,
    Zbot8DofBipedal2Cfg,
    Zbot8DofBipedal3Cfg,
    Zbot8DofBirdCfg,
    Zbot8DofSnakeCfg,
    Zbot8DofVelocityCfg,
    Zbot8DofWheelCfg,
)
from .tasks.direct.zbot_direct.velocity_env_cfg import DEFAULT_VELOCITY_REWARD_SCALES
from .tasks.direct.zbot_direct.bipedal_env import (
    ZbotBipedalEnv,
    ZbotBipedalJointAccEnv,
)
from .tasks.direct.zbot_direct.curriculum_env import (
    Zbot6DofBipedalEnv,
    Zbot6DofQuatEnv,
)
from .tasks.direct.zbot_direct.ground_env import ZbotSnakeEnv, ZbotWheelEnv
from .tasks.direct.zbot_direct.transition_env import Zbot6DofToSnakeEnv
from .tasks.direct.zbot_direct.velocity_env import ZbotVelocityEnv


class Zbot6DofBipedalTaskEnv(Zbot6DofBipedalEnv):
    """6dof bipedal walking with a two-stage reward curriculum."""

    cfg_cls = Zbot6DofBipedalCfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot6DofBipedalQuatTaskEnv(Zbot6DofQuatEnv):
    """6dof bipedal walking; observation includes base_quat_w."""

    cfg_cls = Zbot6DofBipedalQuatCfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot6DofBipedalToSnakeTaskEnv(Zbot6DofToSnakeEnv):
    """6dof transition task from bipedal posture toward snake-like posture."""

    cfg_cls = Zbot6DofBipedalToSnakeCfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot6DofBipedalToSnake1TaskEnv(Zbot6DofToSnakeEnv):
    """6dof transition task variant with forward velocity reward enabled."""

    cfg_cls = Zbot6DofBipedalToSnake1Cfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot6DofVelocityTaskEnv(ZbotVelocityEnv):
    """6dof omnidirectional velocity tracking; play uses keyboard control."""

    cfg_cls = Zbot6DofVelocityCfg
    reward_scales_link = DEFAULT_VELOCITY_REWARD_SCALES
    reward_impl = "tasks/direct/zbot_direct/velocity_rewards.py"


class Zbot6DofVelocityQuatTeacherTaskEnv(ZbotVelocityEnv):
    """6dof velocity teacher task; policy observes privileged base velocity and IMU quaternion."""

    cfg_cls = Zbot6DofVelocityQuatTeacherCfg
    reward_scales_link = DEFAULT_VELOCITY_REWARD_SCALES
    reward_impl = "tasks/direct/zbot_direct/velocity_rewards.py"


class Zbot6DofVelocityImuTaskEnv(ZbotVelocityEnv):
    """6dof velocity student task; policy observes IMU-like signals, not base linear velocity."""

    cfg_cls = Zbot6DofVelocityImuCfg
    reward_scales_link = DEFAULT_VELOCITY_REWARD_SCALES
    reward_impl = "tasks/direct/zbot_direct/velocity_rewards.py"


class Zbot8DofBipedal0TaskEnv(ZbotBipedalEnv):
    """8dof bipedal baseline on the neutral human-like connection."""

    cfg_cls = Zbot8DofBipedal0Cfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot8DofBipedal1TaskEnv(ZbotBipedalJointAccEnv):
    """8dof bipedal variant with joint acceleration regularization."""

    cfg_cls = Zbot8DofBipedal1Cfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot8DofBipedal2TaskEnv(ZbotBipedalJointAccEnv):
    """8dof bipedal variant 2 with joint acceleration regularization."""

    cfg_cls = Zbot8DofBipedal2Cfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot8DofBipedal3TaskEnv(ZbotBipedalJointAccEnv):
    """8dof bipedal variant 3 with joint acceleration regularization."""

    cfg_cls = Zbot8DofBipedal3Cfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot8DofBirdTaskEnv(ZbotBipedalEnv):
    """8dof bird-like morphology using the default bipedal env behavior."""

    cfg_cls = Zbot8DofBirdCfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot8DofSnakeTaskEnv(ZbotSnakeEnv):
    """8dof snake locomotion close to the ground."""

    cfg_cls = Zbot8DofSnakeCfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot8DofWheelTaskEnv(ZbotWheelEnv):
    """8dof wheel-like locomotion with narrower joint target limits."""

    cfg_cls = Zbot8DofWheelCfg
    reward_impl = "tasks/direct/zbot_direct/shared_rewards.py"


class Zbot8DofVelocityTaskEnv(ZbotVelocityEnv):
    """8dof omnidirectional velocity tracking; play uses keyboard control."""

    cfg_cls = Zbot8DofVelocityCfg
    reward_scales_link = ZBOT_8DOF_VELOCITY_REWARD_SCALES
    reward_impl = "tasks/direct/zbot_direct/velocity_rewards.py"
