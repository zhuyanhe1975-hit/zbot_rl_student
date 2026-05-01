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
- To change observations, follow the parent class named by the task below.
- Velocity play tasks use keyboard commands and should run with num_envs=1.
"""

from __future__ import annotations

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


class Zbot6DofBipedalQuatTaskEnv(Zbot6DofQuatEnv):
    """6dof bipedal walking; observation includes base_quat_w."""


class Zbot6DofBipedalToSnakeTaskEnv(Zbot6DofToSnakeEnv):
    """6dof transition task from bipedal posture toward snake-like posture."""


class Zbot6DofBipedalToSnake1TaskEnv(Zbot6DofToSnakeEnv):
    """6dof transition task variant with forward velocity reward enabled."""


class Zbot6DofVelocityTaskEnv(ZbotVelocityEnv):
    """6dof omnidirectional velocity tracking; play uses keyboard control."""


class Zbot6DofVelocityQuatTeacherTaskEnv(ZbotVelocityEnv):
    """6dof velocity teacher task; policy observes privileged base velocity and IMU quaternion."""


class Zbot6DofVelocityImuTaskEnv(ZbotVelocityEnv):
    """6dof velocity student task; policy observes IMU-like signals, not base linear velocity."""


class Zbot8DofBipedal0TaskEnv(ZbotBipedalEnv):
    """8dof bipedal baseline on the neutral human-like connection."""


class Zbot8DofBipedal1TaskEnv(ZbotBipedalJointAccEnv):
    """8dof bipedal variant with joint acceleration regularization."""


class Zbot8DofBipedal2TaskEnv(ZbotBipedalJointAccEnv):
    """8dof bipedal variant 2 with joint acceleration regularization."""


class Zbot8DofBipedal3TaskEnv(ZbotBipedalJointAccEnv):
    """8dof bipedal variant 3 with joint acceleration regularization."""


class Zbot8DofBirdTaskEnv(ZbotBipedalEnv):
    """8dof bird-like morphology using the default bipedal env behavior."""


class Zbot8DofSnakeTaskEnv(ZbotSnakeEnv):
    """8dof snake locomotion close to the ground."""


class Zbot8DofWheelTaskEnv(ZbotWheelEnv):
    """8dof wheel-like locomotion with narrower joint target limits."""


class Zbot8DofVelocityTaskEnv(ZbotVelocityEnv):
    """8dof omnidirectional velocity tracking; play uses keyboard control."""
