from __future__ import annotations

import torch


class ZbotVelocityRewards:
    def _reward_track_lin_vel_xy_exp(self):
        actual_planar_vel = torch.cat((self.base_lin_vel_forward_b, self.base_lin_vel_side_b), dim=1)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - actual_planar_vel), dim=1)
        return torch.exp(-lin_vel_error / 0.05)


    def _reward_track_ang_vel_z_exp(self):
        yaw_rate_error = torch.square(self._commands[:, 2] - self.base_ang_vel_z_b.squeeze(-1))
        return torch.exp(-yaw_rate_error / 0.05)


    def _reward_command_forward_progress(self):
        cmd_xy = self._commands[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        actual_xy = torch.cat((self.base_lin_vel_forward_b, self.base_lin_vel_side_b), dim=1)
        cmd_dir = cmd_xy / torch.clamp(cmd_speed.unsqueeze(-1), min=1.0e-6)
        aligned_speed = torch.sum(actual_xy * cmd_dir, dim=1)
        moving_cmd = cmd_speed > 0.1
        reward = torch.zeros_like(cmd_speed)
        reward[moving_cmd] = torch.clamp(aligned_speed[moving_cmd], min=0.0)
        return reward


    def _reward_command_speed_shortfall(self):
        cmd_xy = self._commands[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        actual_xy = torch.cat((self.base_lin_vel_forward_b, self.base_lin_vel_side_b), dim=1)
        cmd_dir = cmd_xy / torch.clamp(cmd_speed.unsqueeze(-1), min=1.0e-6)
        aligned_speed = torch.sum(actual_xy * cmd_dir, dim=1)
        moving_cmd = cmd_speed > 0.1
        penalty = torch.zeros_like(cmd_speed)
        penalty[moving_cmd] = torch.clamp(cmd_speed[moving_cmd] - aligned_speed[moving_cmd], min=0.0)
        return penalty


    def _reward_lateral_velocity_match(self):
        cmd_y = self._commands[:, 1]
        moving_cmd = torch.abs(cmd_y) > 0.05
        reward = torch.zeros_like(cmd_y)
        reward[moving_cmd] = torch.exp(
            -torch.square(cmd_y[moving_cmd] - self.base_lin_vel_side_b[moving_cmd, 0]) / 0.03
        )
        return reward


    def _reward_yaw_when_translating(self):
        cmd_xy = self._commands[:, :2]
        translating = torch.norm(cmd_xy, dim=1) > 0.15
        low_yaw_cmd = torch.abs(self._commands[:, 2]) < 0.2
        penalty = torch.zeros(self.num_envs, device=self.device)
        mask = translating & low_yaw_cmd
        penalty[mask] = torch.abs(self.base_ang_vel_z_b[mask, 0])
        return penalty


    def _reward_command_stillness(self):
        cmd_xy = self._commands[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        yaw_cmd = torch.abs(self._commands[:, 2])
        actual_speed = torch.norm(torch.cat((self.base_lin_vel_forward_b, self.base_lin_vel_side_b), dim=1), dim=1)
        actual_yaw_rate = torch.abs(self.base_ang_vel_z_b.squeeze(-1))
        joint_motion = torch.mean(torch.abs(self._robot.data.joint_vel), dim=1)
        standing_cmd = (cmd_speed < 0.05) & (yaw_cmd < 0.05)
        penalty = torch.zeros_like(cmd_speed)
        penalty[standing_cmd] = (
            actual_speed[standing_cmd]
            + 0.3 * actual_yaw_rate[standing_cmd]
            + 0.02 * joint_motion[standing_cmd]
        )
        return penalty


    def _reward_lin_vel_xy_variance_l2(self):
        cmd_speed = torch.norm(self._commands[:, :2], dim=1)
        moving_cmd = cmd_speed > 0.1
        actual_planar_vel = torch.cat((self.base_lin_vel_forward_b, self.base_lin_vel_side_b), dim=1)
        vel_delta = actual_planar_vel - self._prev_base_lin_vel
        penalty = torch.zeros(self.num_envs, device=self.device)
        penalty[moving_cmd] = torch.sum(torch.square(vel_delta[moving_cmd]), dim=1)
        return penalty


    def _reward_ang_vel_z_variance_l2(self):
        yaw_cmd = torch.abs(self._commands[:, 2]) > 0.1
        yaw_delta = self.base_ang_vel_z_b.squeeze(-1) - self._prev_base_ang_vel_z
        penalty = torch.zeros(self.num_envs, device=self.device)
        penalty[yaw_cmd] = torch.square(yaw_delta[yaw_cmd])
        return penalty


    def _reward_phase_match(self):
        phase = self._get_step_phase()
        left_should_swing = torch.sin(phase) >= 0.0
        contacts = self.feet_contact_forces > 1.0
        desired_contacts = torch.zeros_like(contacts)
        desired_contacts[:, 0] = ~left_should_swing
        desired_contacts[:, 1] = left_should_swing
        return (desired_contacts == contacts).float().mean(dim=1)


    def _reward_swing_height(self):
        phase = self._get_step_phase()
        left_should_swing = torch.sin(phase) >= 0.0
        swing_height = torch.where(left_should_swing, self.feet_height[:, 0], self.feet_height[:, 1])
        stance_height = torch.where(left_should_swing, self.feet_height[:, 1], self.feet_height[:, 0])
        clearance_target = self.cfg.stepping_clearance_target
        swing_score = 1.0 - torch.abs(swing_height - clearance_target) / max(clearance_target, 1.0e-6)
        swing_score = torch.clamp(swing_score, min=0.0)
        return swing_score * (swing_height > stance_height).float()


    def _reward_leg_toggle(self):
        contacts = self.feet_contact_forces > 1.0
        support_now = -1 * torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        left = contacts[:, 0]
        right = contacts[:, 1]
        support_now[left & ~right] = 0
        support_now[right & ~left] = 1
        changed = (support_now != -1) & (self.last_support_leg != -1) & (support_now != self.last_support_leg)
        reward = changed.float()
        self.last_support_leg = torch.where(support_now != -1, support_now, self.last_support_leg)
        return reward


    def _reward_single_leg_support(self):
        contacts = self.feet_contact_forces > 1.0
        return (torch.sum(contacts.float(), dim=1) == 1).float()


    def _reward_airtime_balance(self):
        return torch.abs(self.feet_air_times[:, 0] - self.feet_air_times[:, 1])


    def _reward_double_support(self):
        contacts = self.feet_contact_forces > 1.0
        return (torch.sum(contacts.float(), dim=1) == 2).float()


    def _reward_dead_time(self):
        moving = torch.norm(self._robot.data.joint_vel, dim=-1) > 1.0
        self.dead_time[moving] = 0.0
        self.dead_time[~moving] += self.step_dt
        self.dead_time = torch.clamp(self.dead_time, 0.0, 5.0)
        return self.dead_time


    def _reward_lin_vel_z_l2(self):
        return torch.square(self.base_lin_vel_b[:, 2])


    def _reward_ang_vel_xy_l2(self):
        return torch.sum(torch.square(self.base_ang_vel_b[:, :2]), dim=1)


    def _reward_action_rate_l2(self):
        return torch.sum(torch.square(self._actions - self._previous_actions), dim=1)


    def _reward_dof_torques_l2(self):
        return torch.sum(torch.square(self._robot.data.applied_torque), dim=1)


    def _reward_joint_deviation_l1(self):
        return torch.sum(torch.abs(self._robot.data.joint_pos - self._joint_reference_pos), dim=1)


    def _reward_feet_forward_bias_integral(self):
        return torch.abs(self._feet_forward_bias_integral)


    def _reward_feet_downward(self):
        return torch.sum(torch.norm(self.feet_z_w[:, :, :2], dim=-1), dim=-1)


    def _reward_feet_forward(self):
        return torch.sum(
            torch.norm(
                self.feet_x_b - self.base_dir_forward_b.unsqueeze(1).repeat(1, len(self.feet_body_idx), 1),
                dim=-1,
            ),
            dim=-1,
        )


    def _reward_feet_air_time(self):
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        return torch.sum(torch.clamp(last_air_time - 0.2, min=0.0) * first_contact, dim=1)


    def _reward_feet_slide(self):
        feet_forces = torch.mean(self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, 2], dim=1)
        contacts = feet_forces > 1.0
        feet_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids, :2]
        return torch.sum(torch.norm(feet_vel, dim=-1) * contacts, dim=1)


    def _reward_flat_orientation_l2(self):
        return torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

