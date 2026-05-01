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
    Zbot8DofRewards,
    ZbotCommonRewards,
    ZbotGroundModeRewards,
)


class ZbotGroundModeEnv(
    ZbotGroundModeRewards, Zbot8DofRewards, ZbotCommonRewards, ZbotBaseEnv
):
    def __init__(self, cfg: ZbotDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._setup_locomotion_buffers(
            torch.full((self.num_envs, 1), 2.0, device=self.device)
        )




    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        self.base_pos_w = self._robot.data.body_pos_w[:, self.base_body_idx].squeeze()
        self.base_quat_w = self._robot.data.body_quat_w[:, self.base_body_idx].squeeze()
        self.feet_quat_w = self._robot.data.body_quat_w[:, self.feet_body_idx]
        self.feet_pos_w = self._robot.data.body_pos_w[:, self.feet_body_idx]

        axis_z = torch.tensor(
            [0, 0, 1], device=self.sim.device, dtype=torch.float32
        ).repeat(
            (self.num_envs, 1)
        )  # base body axis z point to world Y
        self.base_projected_gravity_b = math_utils.quat_apply_inverse(
            self.base_quat_w, self._robot.data.GRAVITY_VEC_W
        )
        self.base_dir_forward_b = torch.cross(self.base_projected_gravity_b, axis_z)
        self.base_dir_forward_w = math_utils.quat_apply(
            self.base_quat_w, self.base_dir_forward_b
        )
        self.base_dir_z_w = math_utils.quat_apply(self.base_quat_w, axis_z)
        self.base_heading_x_err = torch.norm(
            self.base_dir_z_w[..., 1:3], dim=-1
        ).unsqueeze(-1)
        self.head_dir_z_w = math_utils.quat_apply(self.feet_quat_w[:, 0], axis_z)
        self.head_heading_x_err = torch.norm(
            self.head_dir_z_w[..., 1:3], dim=-1
        ).unsqueeze(-1)
        self.base_dir_side_b = axis_z
        self.base_lin_vel_w = self._robot.data.body_lin_vel_w[:, self.base_body_idx, :]
        self.head_lin_vel_w = self._robot.data.body_lin_vel_w[
            :, self.feet_body_idx[0], :
        ]
        base_lin_vel_b = (
            math_utils.quat_apply_inverse(self.base_quat_w, self.base_lin_vel_w)
        ).squeeze()
        self.base_lin_vel_forward_b = (
            torch.sum((base_lin_vel_b * self.base_dir_forward_b), dim=-1)
        ).unsqueeze(-1)
        self.base_lin_vel_side_b = (
            torch.sum((base_lin_vel_b * self.base_dir_side_b), dim=-1)
        ).unsqueeze(-1)
        self.base_ang_vel_w = self._robot.data.body_ang_vel_w[
            :, self.base_body_idx, :
        ].squeeze()
        self.base_ang_vel_z_b = self.base_ang_vel_w[..., 2].unsqueeze(-1)

        axis_x_feet = torch.tensor(
            [1, 0, 0], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 2, 1))
        axis_z_feet = torch.tensor(
            [0, 0, 1], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 2, 1))
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
        self.feet_pos_y_err = (
            self.feet_pos_w[:, :, 1]
            - (self._terrain.env_origins.unsqueeze(1)).repeat(1, 2, 1)[..., 1]
        )
        self.loop_count += 1
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.base_ang_vel_z_b,
                    self.base_projected_gravity_b,
                    self.base_heading_x_err,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    self.joint_speed_limit,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations


class ZbotSnakeEnv(ZbotGroundModeEnv):
    def _termination_terms(self) -> tuple[torch.Tensor, ...]:
        return ()


    def _reward_base_vel_forward(self):
        base_vel_forward = torch.tanh(
            5.0
            * self.cfg.moving_direction
            * self.base_lin_vel_side_b
            / self.joint_speed_limit
        )
        return base_vel_forward.squeeze()

    def _reward_head_vel_x(self):
        head_vel_x = torch.tanh(
            5.0 * (self.cfg.moving_direction * self.head_lin_vel_w[:, 0])
        )
        return head_vel_x.squeeze()


class ZbotWheelEnv(ZbotGroundModeEnv):
    def _pre_physics_step(self, actions: torch.Tensor):
        self._integrate_joint_position_actions(
            actions,
            lower_limit=-0.5 * torch.pi,
            upper_limit=0.5 * torch.pi,
        )

    def _termination_terms(self) -> tuple[torch.Tensor, ...]:
        return (self.robot_com[:, 2] < 0.1,)


    def _reward_com_vel_y(self):
        com_vel_y = torch.mean(self._robot.data.body_lin_vel_w[:, :, 1], dim=1)
        return com_vel_y.squeeze()

    def _reward_com_z(self):
        com_z = torch.abs(self.robot_com[:, 2] - 0.2)
        return com_z.squeeze()
