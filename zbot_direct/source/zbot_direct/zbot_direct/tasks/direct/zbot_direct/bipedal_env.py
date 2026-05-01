# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import isaaclab.utils.math as math_utils

from .base_env import ZbotBaseEnv
from .env_utils import compute_com
from .shared_rewards import (
    Zbot8DofBipedalRewards,
    Zbot8DofRewards,
    ZbotCommonRewards,
)


class ZbotBipedalEnv(Zbot8DofRewards, ZbotCommonRewards, ZbotBaseEnv):

    def __init__(self, cfg: ZbotDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        joint_speed_limit = 0.2 + 1.8 * torch.rand(
            self.num_envs, 1, device=self.device
        )
        self._setup_locomotion_buffers(joint_speed_limit)




    def _get_observations(self) -> dict:
        self._update_locomotion_state()
        obs = torch.cat(
            [tensor for tensor in self._policy_obs_terms() if tensor is not None],
            dim=-1,
        )
        return {"policy": obs}

    def _update_locomotion_state(self) -> None:
        self._previous_actions = self._actions.clone()
        self.base_pos_w = self._robot.data.body_pos_w[:, self.base_body_idx].squeeze()
        self.base_quat_w = self._robot.data.body_quat_w[:, self.base_body_idx].squeeze()
        self.feet_quat_w = self._robot.data.body_quat_w[:, self.feet_body_idx]
        self.feet_pos_w = self._robot.data.body_pos_w[:, self.feet_body_idx]

        axis_z = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.base_projected_gravity_b = math_utils.quat_apply_inverse(
            self.base_quat_w, self._robot.data.GRAVITY_VEC_W
        )
        self.base_dir_forward_b = torch.cross(self.base_projected_gravity_b, axis_z)
        self.base_dir_forward_w = math_utils.quat_apply(
            self.base_quat_w, self.base_dir_forward_b
        )
        self.base_dir_z_w = math_utils.quat_apply(self.base_quat_w, axis_z)
        self.base_heading_x_err = -self.base_dir_forward_w[..., 1].unsqueeze(-1)
        self.base_dir_side_b = axis_z
        self.base_lin_vel_w = self._robot.data.body_lin_vel_w[:, self.base_body_idx, :]
        base_lin_vel_b = math_utils.quat_apply_inverse(
            self.base_quat_w, self.base_lin_vel_w
        ).squeeze()
        self.base_lin_vel_forward_b = torch.sum(
            base_lin_vel_b * self.base_dir_forward_b, dim=-1
        ).unsqueeze(-1)
        self.base_lin_vel_side_b = torch.sum(
            base_lin_vel_b * self.base_dir_side_b, dim=-1
        ).unsqueeze(-1)
        self.base_ang_vel_w = self._robot.data.body_ang_vel_w[:, self.base_body_idx, :].squeeze()
        self.base_ang_vel_z_b = self.base_ang_vel_w[..., 2].unsqueeze(-1)

        axis_x_feet = torch.tensor([1, 0, 0], device=self.sim.device, dtype=torch.float32).repeat(
            (self.num_envs, 2, 1)
        )
        axis_z_feet = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat(
            (self.num_envs, 2, 1)
        )
        self.feet_z_w = math_utils.quat_apply(self.feet_quat_w, axis_z_feet)
        self.feet_x_w = math_utils.quat_apply(self.feet_quat_w, axis_x_feet)
        self.feet_x_b = math_utils.quat_apply_inverse(
            self.base_quat_w.repeat(1, 2, 1), self.feet_x_w
        )
        self.speed_ratio = np.clip(
            float(
                torch.tanh(
                    5.0
                    * torch.mean(
                        self.cfg.moving_direction * self.base_lin_vel_forward_b, dim=0
                    )
                )
            ),
            0,
            1,
        )
        body_com = self._robot.data.body_com_pos_w
        body_mass = self._robot.data.default_mass.to(device="cuda")
        self.robot_com = compute_com(body_com, body_mass)
        feet_com_vec = self.feet_pos_w - self.robot_com.unsqueeze(1).repeat(1, 2, 1)
        self.feet_com_distance = torch.norm(feet_com_vec[:, :, :2], dim=-1).squeeze()
        self.base_pos_y_err = self.base_pos_w[:, 1] - self._terrain.env_origins[:, 1]
        self.loop_count += 1

    def _policy_obs_terms(self) -> tuple[torch.Tensor | None, ...]:
        return (
            self.base_ang_vel_z_b,
            self.base_projected_gravity_b,
            self.base_heading_x_err,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            self._actions,
            self.joint_speed_limit,
        )


class ZbotBipedalJointAccEnv(Zbot8DofBipedalRewards, ZbotBipedalEnv):
    pass
