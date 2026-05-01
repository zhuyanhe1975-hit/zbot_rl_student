from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

from isaaclab.envs import DirectRLEnv

from .base_env import ZbotBaseEnv
from .velocity_commands import ZbotVelocityCommands
from .velocity_debug_vis import ZbotVelocityDebugVis
from .velocity_rewards import ZbotVelocityRewards


class ZbotVelocityEnv(
    ZbotVelocityRewards, ZbotVelocityCommands, ZbotVelocityDebugVis, ZbotBaseEnv
):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._manual_commands = torch.zeros_like(self._commands)
        self._manual_command_override_enabled = False
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        self._joint_pos_target_delta = torch.zeros_like(self._robot.data.default_joint_pos)
        self._prev_base_lin_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_base_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        self.joint_speed_limit = 2.0 + 0.0 * torch.rand(self.num_envs, 1, device=self.device)
        self._step_frequency_cmd = torch.zeros(self.num_envs, device=self.device)
        self._step_phase_offset = torch.zeros(self.num_envs, device=self.device)
        self._feet_forward_bias_integral = torch.zeros(self.num_envs, device=self.device)
        self._joint_reference_pos = self._make_joint_reference_pos()
        self._joint_target_lower = self._make_optional_joint_limit("joint_target_lower")
        self._joint_target_upper = self._make_optional_joint_limit("joint_target_upper")

        self._setup_body_indices()
        self.dead_time = torch.zeros(self.num_envs, device=self.device)
        self.last_support_leg = -1 * torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self._setup_reward_functions()

        self._goal_vel_visualizer = None
        self._current_vel_visualizer = None
        self._goal_yaw_visualizer = None
        self._current_yaw_visualizer = None

    def _make_joint_reference_pos(self) -> torch.Tensor:
        if hasattr(self.cfg, "reference_joint_pos"):
            reference = torch.tensor(self.cfg.reference_joint_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
            return reference.repeat(self.num_envs, 1)
        return self._robot.data.default_joint_pos.clone()

    def _make_optional_joint_limit(self, cfg_name: str) -> torch.Tensor | None:
        if not hasattr(self.cfg, cfg_name):
            return None
        return torch.tensor(getattr(self.cfg, cfg_name), dtype=torch.float32, device=self.device).unsqueeze(0)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._update_state_buffers()
        self._prev_base_lin_vel[:, 0].copy_(self.base_lin_vel_forward_b.squeeze(-1))
        self._prev_base_lin_vel[:, 1].copy_(self.base_lin_vel_side_b.squeeze(-1))
        self._prev_base_ang_vel_z.copy_(self.base_ang_vel_z_b.squeeze(-1))
        if self._manual_command_override_enabled:
            self._commands.copy_(self._manual_commands)
            self._command_time_left.zero_()
        else:
            self._resample_commands((self._command_time_left <= 0.0).nonzero(as_tuple=False).flatten())
            self._command_time_left -= self.step_dt

        self._actions = torch.tanh(actions)
        self._joint_pos_target_delta += (
            torch.pi * self.joint_speed_limit * self.cfg.action_scale * self._actions * self.step_dt
        )
        self._joint_pos_target_delta = torch.clamp(self._joint_pos_target_delta, -torch.pi, torch.pi)
        unclamped_joint_targets = self._joint_reference_pos + self._joint_pos_target_delta
        if self._joint_target_lower is not None and self._joint_target_upper is not None:
            self._processed_actions = torch.clamp(
                unclamped_joint_targets, self._joint_target_lower, self._joint_target_upper
            )
        else:
            self._processed_actions = unclamped_joint_targets
        self._joint_pos_target_delta = self._processed_actions - self._joint_reference_pos

    def _get_observations(self) -> dict:
        self._update_state_buffers()
        self._previous_actions.copy_(self._actions)

        obs = torch.cat(
            (
                self.base_lin_vel_b,
                self.base_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._get_step_command_obs(),
                self._robot.data.joint_pos - self._joint_reference_pos,
                self._robot.data.joint_vel,
                self._actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._update_state_buffers()
        self._update_feet_forward_bias_integral()
        stepping_factor, velocity_factor = self._get_curriculum_factors()
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for name, reward_func in self.reward_functions.items():
            reward_factor = self._get_reward_factor(name, stepping_factor, velocity_factor)
            value = reward_func() * self.reward_scales[name] * reward_factor
            reward += value
            self._episode_sums[name] += value

        terminated_ids = self.reset_terminated.nonzero(as_tuple=False).flatten()
        reward[terminated_ids] -= 10.0
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_state_buffers()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        low_height = (self._robot.data.body_pos_w[:, self.base_body_idx, 2] < self.cfg.termination_height).squeeze(-1)
        bad_orientation = (
            torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > self.cfg.termination_gravity_xy
        )
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        undesired_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0]
            > self.cfg.non_foot_contact_force_threshold,
            dim=1,
        )
        feet_separation = torch.norm(self.feet_pos_w[:, 0, :2] - self.feet_pos_w[:, 1, :2], dim=1)
        collapsed_posture = feet_separation < self.cfg.min_feet_separation
        return low_height | bad_orientation | undesired_contact | collapsed_posture, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._manual_commands[env_ids] = 0.0
        self._joint_pos_target_delta[env_ids] = 0.0
        self._prev_base_lin_vel[env_ids] = 0.0
        self._prev_base_ang_vel_z[env_ids] = 0.0
        self._feet_forward_bias_integral[env_ids] = 0.0
        self.dead_time[env_ids] = 0.0
        self.last_support_leg[env_ids] = -1

        joint_pos = self._joint_reference_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._write_robot_state(env_ids, joint_pos, joint_vel)

        if self._manual_command_override_enabled:
            self._commands[env_ids] = self._manual_commands[env_ids]
            self._command_time_left[env_ids] = 0.0
        else:
            self._resample_commands(env_ids)
        self._resample_step_commands(env_ids)
        self._update_state_buffers()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict(extras)
        self.extras["log"]["Episode_Termination/fall"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        stepping_factor, velocity_factor = self._get_curriculum_factors()
        xy_scale, yaw_scale = self._get_command_curriculum_scales(velocity_factor)
        self.extras["log"]["Curriculum/stepping_factor"] = stepping_factor
        self.extras["log"]["Curriculum/velocity_factor"] = velocity_factor
        self.extras["log"]["Curriculum/xy_command_scale"] = xy_scale
        self.extras["log"]["Curriculum/yaw_command_scale"] = yaw_scale
        self.extras["log"]["Learning/command_abs_vx"] = torch.mean(torch.abs(self._commands[:, 0])).item()
        self.extras["log"]["Learning/command_abs_vy"] = torch.mean(torch.abs(self._commands[:, 1])).item()
        self.extras["log"]["Learning/command_abs_wz"] = torch.mean(torch.abs(self._commands[:, 2])).item()
        self.extras["log"]["Learning/command_speed_xy"] = torch.mean(torch.norm(self._commands[:, :2], dim=1)).item()
        actual_planar_vel = torch.cat((self.base_lin_vel_forward_b, self.base_lin_vel_side_b), dim=1)
        self.extras["log"]["Learning/actual_speed_xy"] = torch.mean(torch.norm(actual_planar_vel, dim=1)).item()
        self.extras["log"]["Learning/feet_forward_bias_integral_abs"] = torch.mean(
            torch.abs(self._feet_forward_bias_integral)
        ).item()
        self.extras["log"]["Learning/vel_tracking_error_xy"] = torch.mean(
            torch.norm(self._commands[:, :2] - actual_planar_vel, dim=1)
        ).item()
        self.extras["log"]["Learning/vel_tracking_error_yaw"] = torch.mean(
            torch.abs(self._commands[:, 2] - self.base_ang_vel_z_b.squeeze(-1))
        ).item()



    def _update_state_buffers(self):
        self.base_pos_w = self._robot.data.body_pos_w[:, self.base_body_idx].squeeze(1)
        self.base_quat_w = self._robot.data.body_quat_w[:, self.base_body_idx].squeeze(1)
        self.feet_quat_w = self._robot.data.body_quat_w[:, self.feet_body_idx]
        axis_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        axis_x_feet = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, len(self.feet_body_idx), 1
        )
        axis_z_feet = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            self.num_envs, len(self.feet_body_idx), 1
        )
        self.base_projected_gravity_b = math_utils.quat_apply_inverse(self.base_quat_w, self._robot.data.GRAVITY_VEC_W)
        self.base_dir_forward_b = torch.cross(self.base_projected_gravity_b, axis_z, dim=-1)
        self.base_dir_forward_b = torch.nn.functional.normalize(self.base_dir_forward_b, dim=-1)
        self.base_dir_side_b = axis_z
        self.base_dir_forward_w = math_utils.quat_apply(self.base_quat_w, self.base_dir_forward_b)
        self.base_dir_side_w = torch.cross(axis_z, self.base_dir_forward_w, dim=-1)
        self.base_dir_side_w = torch.nn.functional.normalize(self.base_dir_side_w, dim=-1)
        self.feet_z_w = math_utils.quat_apply(self.feet_quat_w, axis_z_feet)
        self.feet_x_w = math_utils.quat_apply(self.feet_quat_w, axis_x_feet)
        base_quat_for_feet = self.base_quat_w.unsqueeze(1).repeat(1, len(self.feet_body_idx), 1)
        self.feet_x_b = math_utils.quat_apply_inverse(base_quat_for_feet, self.feet_x_w)
        self.base_lin_vel_w = self._robot.data.body_lin_vel_w[:, self.base_body_idx, :].squeeze(1)
        self.base_lin_vel_b = math_utils.quat_apply_inverse(self.base_quat_w, self.base_lin_vel_w)
        self.base_lin_vel_forward_b = torch.sum(self.base_lin_vel_b * self.base_dir_forward_b, dim=-1, keepdim=True)
        self.base_lin_vel_side_b = torch.sum(self.base_lin_vel_b * self.base_dir_side_b, dim=-1, keepdim=True)
        self.base_ang_vel_w = self._robot.data.body_ang_vel_w[:, self.base_body_idx, :].squeeze(1)
        self.base_ang_vel_b = math_utils.quat_apply_inverse(self.base_quat_w, self.base_ang_vel_w)
        self.base_ang_vel_z_b = self.base_ang_vel_w[:, 2:3]
        self.feet_pos_w = self._robot.data.body_pos_w[:, self.feet_body_idx]
        self.feet_pos_rel_base_w = self.feet_pos_w - self.base_pos_w.unsqueeze(1)
        self.feet_forward_pos_b = torch.sum(
            self.feet_pos_rel_base_w * self.base_dir_forward_w.unsqueeze(1),
            dim=-1,
        )
        self.feet_forward_diff = self.feet_forward_pos_b[:, 0] - self.feet_forward_pos_b[:, 1]
        self.feet_height = self.feet_pos_w[:, :, 2] - self._terrain.env_origins[:, 2].unsqueeze(-1)
        self.feet_contact_forces = torch.mean(
            self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, 2], dim=1
        )
        self.feet_air_times = self._contact_sensor.data.last_air_time[:, self._feet_ids]

    def _update_feet_forward_bias_integral(self):
        self._feet_forward_bias_integral += self.feet_forward_diff * self.step_dt
        self._feet_forward_bias_integral = torch.clamp(self._feet_forward_bias_integral, -5.0, 5.0)

    def _get_step_phase(self) -> torch.Tensor:
        return 2.0 * torch.pi * self._step_frequency_cmd * (self.episode_length_buf.float() * self.step_dt) + self._step_phase_offset

    def _get_step_command_obs(self) -> torch.Tensor:
        phase = self._get_step_phase()
        freq_min, freq_max = self.cfg.stepping_frequency_range
        freq_norm = 2.0 * (self._step_frequency_cmd - freq_min) / max(freq_max - freq_min, 1.0e-6) - 1.0
        return torch.stack((torch.sin(phase), torch.cos(phase), freq_norm), dim=-1)







































ZbotDirectEnv = ZbotVelocityEnv
