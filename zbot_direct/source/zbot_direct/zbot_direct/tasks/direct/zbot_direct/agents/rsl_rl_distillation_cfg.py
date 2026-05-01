# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)
from rsl_rl.algorithms import Distillation


class TeacherActionDistillation(Distillation):
    def __init__(
        self,
        *args,
        teacher_action_warmup_updates: int = 500,
        teacher_action_blend_updates: int = 500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_action_warmup_updates = teacher_action_warmup_updates
        self.teacher_action_blend_updates = max(1, teacher_action_blend_updates)
        self._last_teacher_action_ratio = 1.0

    def act(self, obs):
        student_actions = self.student(obs, stochastic_output=True).detach()
        teacher_actions = self.teacher(obs).detach()
        self.transition.privileged_actions = teacher_actions

        teacher_ratio = self._get_teacher_action_ratio()
        self._last_teacher_action_ratio = teacher_ratio
        rollout_actions = teacher_ratio * teacher_actions + (1.0 - teacher_ratio) * student_actions
        self.transition.actions = rollout_actions
        self.transition.observations = obs
        return rollout_actions

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict["teacher_action_ratio"] = self._last_teacher_action_ratio
        return loss_dict

    def _get_teacher_action_ratio(self) -> float:
        if self.num_updates < self.teacher_action_warmup_updates:
            return 1.0
        blend_progress = (self.num_updates - self.teacher_action_warmup_updates) / self.teacher_action_blend_updates
        return max(0.0, min(1.0, 1.0 - blend_progress))


@configclass
class TeacherActionDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = (
        "zbot_direct.tasks.direct.zbot_direct.agents.rsl_rl_distillation_cfg:TeacherActionDistillation"
    )
    teacher_action_warmup_updates = 500
    teacher_action_blend_updates = 500


@configclass
class VelocityImuDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "zbot_direct"
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = TeacherActionDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
    )
