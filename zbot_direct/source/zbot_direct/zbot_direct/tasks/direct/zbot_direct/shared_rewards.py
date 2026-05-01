# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.utils.math as math_utils


class ZbotCommonRewards:
    def _reward_action_rate(self):
            # action rate
            action_rate = torch.sum(
                torch.square(self._actions - self._previous_actions), dim=1
            )
            return action_rate

    def _reward_airtime_balance(self):
            airtime_balance = torch.abs(
                self.feet_air_times[:, 0] - self.feet_air_times[:, 1]
            )
            return airtime_balance

    def _reward_airtime_sum(self):
            airtime_sum = torch.tanh(torch.sum(self.feet_air_times, dim=-1))
            return airtime_sum

    def _reward_base_ang_vel_z(self):
            return torch.abs(self.base_ang_vel_b.squeeze())

    def _reward_base_heading_x(self):
            return torch.abs(self.base_heading_x_err.squeeze())

    def _reward_base_heading_x_sum(self):
            self.base_heading_x_sum += 0.01 * (self.base_heading_x_err)
            self.base_heading_x_sum = torch.clip(self.base_heading_x_sum, -1, 1)
            return torch.abs(self.base_heading_x_sum.squeeze())

    def _reward_base_pos_y_err(self):
            y_err = 10.0*(torch.abs(self.feet_pos_w[:,0, 1] + self.feet_pos_w[:,1, 1] - 2.0*self._terrain.env_origins[:,1]) + torch.abs(self.base_pos_w[:,1] - self._terrain.env_origins[:,1]))
            return y_err

    def _reward_base_vel_side(self):
            return torch.abs(self.base_lin_vel_side_b.squeeze())

    def _reward_com_in_feet(self):
            com_in_feet = 10.0 * torch.min(self.feet_com_distance[:, 0], dim=-1)[0]
            return com_in_feet

    def _reward_dead_time(self):
            mask_0 = torch.norm(self._robot.data.joint_vel, dim=-1) < 1.0
            mask_1 = torch.logical_not(mask_0)
            self.dead_time[mask_1] = 0
            self.dead_time[mask_0] += self.step_dt
            self.dead_time = torch.clip(self.dead_time, 0, 10)
            return self.dead_time

    def _reward_feet_distance_side_b(self):
            feet_disvec_w = self.feet_pos_w[:, 0, :] - self.feet_pos_w[:, 1, :]
            feet_disvec_w[..., 2] = 0.0
            feet_disvec_b = math_utils.quat_apply_inverse(self.base_quat_w, feet_disvec_w)
            feet_distance_b = torch.abs(
                torch.abs(torch.sum(feet_disvec_b * self.base_dir_side_b, dim=-1)) - 0.12
            )
            return feet_distance_b

    def _reward_feet_downward(self):
            feet_downward = torch.sum(torch.norm(self.feet_z_w[:, :, :2], dim=-1), dim=-1)
            return feet_downward

    def _reward_feet_force_diff(self):
            feet_force_diff = (
                1.0
                * torch.sign(self.feet_force_sum)
                * (self.feet_contact_forces[:, 1] - self.feet_contact_forces[:, 0])
            )

            return feet_force_diff

    def _reward_feet_force_max(self):
            feet_force_max = torch.max(self.feet_contact_forces, dim=-1)[0]

            return feet_force_max

    def _reward_feet_force_sum(self):
            self.feet_force_sum += 0.001 * (
                self.feet_contact_forces[:, 0] - self.feet_contact_forces[:, 1]
            )

            return torch.abs(self.feet_force_sum)

    def _reward_feet_forward(self):
            # Reward z axis base linear velocity
            feet_forward = torch.sum(
                torch.norm(
                    (self.feet_x_b - self.base_dir_forward_b.unsqueeze(1).repeat(1, 2, 1)),
                    dim=-1,
                ),
                dim=-1,
            )
            return feet_forward

    def _reward_feet_slide(self):
            """Penalize feet sliding.

            This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
            norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
            agent is penalized only when the feet are in contact with the ground.
            """
            # Penalize feet sliding
            contacts = self.feet_contact_forces > 1.0
            feet_vel = self._robot.data.body_lin_vel_w[:, self.feet_body_idx, :2]
            feet_slide = 1.0 * torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)
            return feet_slide

    def _reward_not_single_leg(self):
            contacts = self.feet_contact_forces > 1.0
            not_single_leg = 1.0 - 1.0 * (torch.sum(1.0 * contacts, dim=1) == 1)
            return not_single_leg

    def _reward_shape_symmetry(self):
            jp = self.p_delta
            symmetry_err = torch.abs(jp[:,0] - jp[:,5]) + torch.abs(jp[:,1] - jp[:,4]) + torch.abs(jp[:,2] - jp[:,3])
            return symmetry_err

    def _reward_similar_to_default(self):
            return torch.sum(
                torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos),
                dim=1,
            )

    def _reward_single_leg(self):
            contacts = self.feet_contact_forces > 1.0
            single_leg = 1.0 * (torch.sum(1.0 * contacts, dim=1) == 1)
            return single_leg

    def _reward_torques(self):
            joint_torques = 0.002*torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
            return joint_torques

    def _reward_undesired_contact(self):
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            is_contact = (
                torch.max(
                    torch.norm(
                        net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1
                    ),
                    dim=1,
                )[0]
                > 1.0
            )
            contacts = torch.sum(is_contact, dim=1)
            return contacts


class Zbot6DofRewards:
    def _reward_base_vel_forward(self):
            base_vel_forward = torch.tanh(
                10.0
                * self.cfg.moving_direction
                * self.base_lin_vel_forward_b
                * (self.base_pos_w[:, 2] < 0.06).unsqueeze(-1) )
            return base_vel_forward.squeeze()

    def _reward_step_length(self):
            # Reward z axis base linear velocity

            force_c = 10.0
            # feet_just_down
            feet_down_idx = torch.logical_and(
                (self.feet_contact_forces > force_c),
                (self.feet_contact_forces_last < force_c),
            )  # feet_just_down





            feet_step_vec_w = self.feet_pos_w - self.feet_down_pos_last
            feet_step_vec_b = math_utils.quat_apply_inverse(
                self.base_quat_w.repeat(1, 2, 1), feet_step_vec_w
            )
            feet_step_length_b = torch.sum(
                feet_step_vec_b * (self.base_dir_forward_b.unsqueeze(1).repeat(1, 2, 1)),
                dim=-1,
            ).squeeze()
            self.feet_step_length[feet_down_idx] = feet_step_length_b[feet_down_idx]


            rew_feet_step_length = (
                self.cfg.moving_direction * torch.min(self.feet_step_length, dim=-1)[0]
            )



            self.feet_down_pos_last[feet_down_idx, :] = self.feet_pos_w[feet_down_idx, :]
            self.feet_contact_forces_last[:] = self.feet_contact_forces[:]  # refresh last
            return torch.tanh(15.0 * rew_feet_step_length)

    def _reward_base_height(self):
            base_height = self.base_pos_w[:, 2] - self._terrain.env_origins[:, 2] - 0.25
            return base_height


class Zbot8DofRewards:
    def _reward_base_vel_forward(self):
            base_vel_forward = torch.tanh(
                5.0
                * self.cfg.moving_direction
                * self.base_lin_vel_forward_b
                / self.joint_speed_limit
            )
            return base_vel_forward.squeeze()

    def _reward_step_length(self):
            # Reward z axis base linear velocity

            force_c = 10.0
            # feet_just_down
            feet_down_idx = torch.logical_and(
                (self.feet_contact_forces > force_c),
                (self.feet_contact_forces_last < force_c),
            )  # feet_just_down





            feet_step_vec_w = self.feet_pos_w - self.feet_down_pos_last
            feet_step_vec_b = math_utils.quat_apply_inverse(
                self.base_quat_w.repeat(1, 2, 1), feet_step_vec_w
            )
            feet_step_length_b = (
                self.cfg.moving_direction
                * torch.sum(
                    feet_step_vec_b
                    * (self.base_dir_forward_b.unsqueeze(1).repeat(1, 2, 1)),
                    dim=-1,
                ).squeeze()
            )
            self.feet_step_length[feet_down_idx] = feet_step_length_b[feet_down_idx]


            rew_feet_step_length = torch.min(self.feet_step_length, dim=-1)[0]



            self.feet_down_pos_last[feet_down_idx, :] = self.feet_pos_w[feet_down_idx, :]
            self.feet_contact_forces_last[:] = self.feet_contact_forces[:]  # refresh last
            return 5.0 * rew_feet_step_length


class Zbot8DofBipedalRewards:
    def _reward_joint_acc(self):
            joint_accel = 0.001 * torch.sum(
                torch.square(self._robot.data.joint_acc), dim=1
            )
            return joint_accel


class ZbotGroundModeRewards:
    def _reward_base_vel_x(self):
            base_vel_x = torch.tanh(
                2.0
                * self.cfg.moving_direction
                * self.base_lin_vel_w[..., 0]
                / self.joint_speed_limit
            )
            return base_vel_x.squeeze()

    def _reward_close_ground(self):
            return torch.tanh(-50.0 * (self.robot_com[:, 2] - 0.04))

    def _reward_head_heading_x(self):
            return torch.abs(self.head_heading_x_err.squeeze())

    def _reward_feet_pos_y_err(self):
            y_err = 10.0 * (
                torch.abs(self.feet_pos_w[:, 0, 1] - self._terrain.env_origins[:, 1])
                + torch.abs(self.feet_pos_w[:, 1, 1] - self._terrain.env_origins[:, 1])
            )
            return y_err

    def _reward_joint_acc(self):
            joint_accel = 0.0001 * torch.sum(
                torch.square(self._robot.data.joint_acc), dim=1
            )
            return joint_accel
