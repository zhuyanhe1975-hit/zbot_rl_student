# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level ZBot task package.

Student-facing files:
- env.py: task environment overview
- cfg.py: task configuration overview

Importing this package registers all Gym environments.
"""

import gymnasium as gym
from gymnasium.envs.registration import registry


_TASKS = (
    (
        "Zbot-Direct-6dof-bipedal-v0",
        "Zbot6DofBipedalTaskEnv",
        "Zbot6DofBipedalCfg",
    ),
    (
        "Zbot-Direct-6dof-bipedal-velocity-v0",
        "Zbot6DofVelocityTaskEnv",
        "Zbot6DofVelocityCfg",
    ),
    (
        "Zbot-Direct-6dof-bipedal-velocity-imu-v0",
        "Zbot6DofVelocityImuTaskEnv",
        "Zbot6DofVelocityImuCfg",
        "zbot_direct.tasks.direct.zbot_direct.agents.rsl_rl_distillation_cfg:VelocityImuDistillationRunnerCfg",
    ),
    (
        "Zbot-Direct-6dof-bipedal-quat-v0",
        "Zbot6DofBipedalQuatTaskEnv",
        "Zbot6DofBipedalQuatCfg",
    ),
    (
        "Zbot-Direct-6dof-bipedal-to-snake-v0",
        "Zbot6DofBipedalToSnakeTaskEnv",
        "Zbot6DofBipedalToSnakeCfg",
    ),
    (
        "Zbot-Direct-6dof-bipedal-to-snake-v1",
        "Zbot6DofBipedalToSnake1TaskEnv",
        "Zbot6DofBipedalToSnake1Cfg",
    ),
    (
        "Zbot-Direct-8dof-bipedal-v0",
        "Zbot8DofBipedal0TaskEnv",
        "Zbot8DofBipedal0Cfg",
    ),
    (
        "Zbot-Direct-8dof-bipedal-v1",
        "Zbot8DofBipedal1TaskEnv",
        "Zbot8DofBipedal1Cfg",
    ),
    (
        "Zbot-Direct-8dof-bipedal-v2",
        "Zbot8DofBipedal2TaskEnv",
        "Zbot8DofBipedal2Cfg",
    ),
    (
        "Zbot-Direct-8dof-bipedal-v3",
        "Zbot8DofBipedal3TaskEnv",
        "Zbot8DofBipedal3Cfg",
    ),
    (
        "Zbot-Direct-8dof-bird-v0",
        "Zbot8DofBirdTaskEnv",
        "Zbot8DofBirdCfg",
    ),
    (
        "Zbot-Direct-8dof-snake-v0",
        "Zbot8DofSnakeTaskEnv",
        "Zbot8DofSnakeCfg",
    ),
    (
        "Zbot-Direct-8dof-wheel-v0",
        "Zbot8DofWheelTaskEnv",
        "Zbot8DofWheelCfg",
    ),
    (
        "Zbot-Direct-8dof-bipedal-velocity-v0",
        "Zbot8DofVelocityTaskEnv",
        "Zbot8DofVelocityCfg",
    ),
)


def register_tasks() -> None:
    for task in _TASKS:
        task_id, env_class, cfg_class = task[:3]
        agent_cfg = (
            task[3]
            if len(task) > 3
            else "zbot_direct.tasks.direct.zbot_direct.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
        )
        if task_id in registry:
            continue
        gym.register(
            id=task_id,
            entry_point=f"zbot_direct.env:{env_class}",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"zbot_direct.cfg:{cfg_class}",
                "rsl_rl_cfg_entry_point": agent_cfg,
            },
        )


register_tasks()
