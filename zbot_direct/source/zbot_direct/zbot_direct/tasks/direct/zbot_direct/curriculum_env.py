# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from .bipedal_env import ZbotBipedalEnv
from .shared_rewards import Zbot6DofRewards


class ZbotCurriculumRewards:
    @staticmethod
    def _ordered_reward_names(*reward_scale_dicts: dict[str, float]) -> list[str]:
        names: list[str] = []
        for reward_scale_dict in reward_scale_dicts:
            for name in reward_scale_dict.keys():
                if name not in names:
                    names.append(name)
        return names

    def _init_reward_functions(self, cfg) -> None:
        self.curriculum_cfg = getattr(cfg, "curriculum_cfg", None)
        self.curriculum_stage = 1
        self.curriculum_last_metric = 0.0
        self.curriculum_promotion_counter = 0

        if not self.curriculum_cfg:
            return super()._init_reward_functions(cfg)

        self.stage_reward_scales = {
            1: self.curriculum_cfg["stage_1_reward_scales"],
            2: self.curriculum_cfg["stage_2_reward_scales"],
        }
        self.curriculum_promotion_metric = self.curriculum_cfg["promotion_metric"]
        self.curriculum_promotion_threshold = self.curriculum_cfg["promotion_threshold"]
        self.curriculum_promotion_window = self.curriculum_cfg["promotion_window_steps"]

        reward_names = self._ordered_reward_names(*self.stage_reward_scales.values())
        self.reward_functions = {
            name: getattr(self, "_reward_" + name) for name in reward_names
        }
        self._episode_sums = {
            name: self._new_env_float_buffer() for name in reward_names
        }
        self._set_active_reward_stage(1)

    def _set_active_reward_stage(self, stage: int) -> None:
        self.curriculum_stage = stage
        self.reward_scales = {
            name: scale * self.step_dt
            for name, scale in self.stage_reward_scales[stage].items()
        }

    def _maybe_promote_curriculum(self, raw_rewards: dict[str, torch.Tensor]) -> None:
        if not self.curriculum_cfg or self.curriculum_stage != 1:
            return

        metric_value = float(
            torch.mean(raw_rewards[self.curriculum_promotion_metric]).item()
        )
        self.curriculum_last_metric = metric_value

        if metric_value > self.curriculum_promotion_threshold:
            self.curriculum_promotion_counter += 1
        else:
            self.curriculum_promotion_counter = 0

        if self.curriculum_promotion_counter < self.curriculum_promotion_window:
            return

        self._set_active_reward_stage(2)
        print(
            "[INFO] Curriculum promoted to stage 2 "
            f"(metric={metric_value:.4f}, threshold={self.curriculum_promotion_threshold:.4f})."
        )

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        raw_rewards: dict[str, torch.Tensor] = {}

        for name, scale in self.reward_scales.items():
            raw_rewards[name] = self.reward_functions[name]()
            rew = raw_rewards[name] * scale
            reward += rew
            self._episode_sums[name] += rew

        if self.curriculum_cfg and self.curriculum_promotion_metric not in raw_rewards:
            raw_rewards[self.curriculum_promotion_metric] = self.reward_functions[
                self.curriculum_promotion_metric
            ]()
        self._maybe_promote_curriculum(raw_rewards)

        terminated_ids = self.reset_terminated.nonzero(as_tuple=False).squeeze(-1)
        reward[terminated_ids] -= 20.0
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        extras = self.extras.setdefault("log", {})
        extras["Curriculum/stage"] = float(self.curriculum_stage)
        if self.curriculum_cfg:
            extras["Curriculum/feet_force_diff_mean"] = self.curriculum_last_metric
            extras["Curriculum/promotion_threshold"] = float(
                self.curriculum_promotion_threshold
            )


class Zbot6DofBipedalEnv(ZbotCurriculumRewards, Zbot6DofRewards, ZbotBipedalEnv):
    def _reward_base_vel_forward(self):
        base_vel_forward = torch.tanh(
            10.0
            * self.cfg.moving_direction
            * self.base_lin_vel_forward_b
            / self.joint_speed_limit
        )
        return base_vel_forward.squeeze()


class Zbot6DofQuatEnv(Zbot6DofRewards, ZbotBipedalEnv):
    def _policy_obs_terms(self) -> tuple[torch.Tensor | None, ...]:
        return (
            self.base_lin_vel_forward_b,
            self.base_quat_w,
            self.base_ang_vel_w,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            self._actions,
            self.joint_speed_limit,
        )

    def _reward_base_vel_forward(self):
        v = (self.cfg.moving_direction * self.base_lin_vel_forward_b).squeeze()
        v_target = 0.5
        sigma = 0.3
        return torch.exp(-((v - v_target) / sigma) ** 2)
