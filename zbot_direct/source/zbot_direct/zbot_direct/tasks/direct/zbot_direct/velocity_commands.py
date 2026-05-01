from __future__ import annotations

import torch


class ZbotVelocityCommands:
    _VELOCITY_REWARD_NAMES = {
        "track_lin_vel_xy_exp",
        "command_forward_progress",
        "command_speed_shortfall",
        "lateral_velocity_match",
        "yaw_when_translating",
        "command_stillness",
        "yaw_acc_z_l2",
        "lin_acc_xy_l2",
    }
    _STEPPING_REWARD_NAMES = {
        "phase_match",
        "swing_height",
        "leg_toggle",
        "single_leg_support",
        "airtime_balance",
        "double_support",
        "dead_time",
    }

    def _resample_commands(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        _, velocity_factor = self._get_curriculum_factors()
        xy_scale, yaw_scale = self._get_command_curriculum_scales(velocity_factor)
        ranges = self.cfg.command_ranges
        self._commands[env_ids] = 0.0
        if velocity_factor <= 0.0:
            pass
        elif velocity_factor < self.cfg.yaw_tracking_start_ratio:
            self._commands[env_ids, 0] = self._sample_uniform(
                (ranges["lin_vel_x"][0] * xy_scale, ranges["lin_vel_x"][1] * xy_scale), env_ids.numel()
            )
            self._commands[env_ids, 1] = self._sample_uniform(
                (ranges["lin_vel_y"][0] * xy_scale, ranges["lin_vel_y"][1] * xy_scale), env_ids.numel()
            )
        else:
            self._commands[env_ids, 0] = self._sample_uniform(ranges["lin_vel_x"], env_ids.numel())
            self._commands[env_ids, 1] = self._sample_uniform(ranges["lin_vel_y"], env_ids.numel())
            self._commands[env_ids, 2] = self._sample_uniform(
                (ranges["ang_vel_z"][0] * yaw_scale, ranges["ang_vel_z"][1] * yaw_scale), env_ids.numel()
            )

        standing_mask = torch.rand(env_ids.numel(), device=self.device) < self.cfg.standing_probability
        self._commands[env_ids[standing_mask]] = 0.0

        min_time, max_time = self.cfg.command_resample_time_range
        self._command_time_left[env_ids] = self._sample_uniform((min_time, max_time), env_ids.numel())


    def _resample_step_commands(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self._step_frequency_cmd[env_ids] = self._sample_uniform(self.cfg.stepping_frequency_range, env_ids.numel())
        self._step_phase_offset[env_ids] = self._sample_uniform((0.0, 2.0 * torch.pi), env_ids.numel())
        self._step_phase_cache_step = -1


    def _get_curriculum_factors(self) -> tuple[float, float]:
        warmup_steps = max(1, int(self.cfg.curriculum_warmup_steps))
        transition_steps = max(1, int(self.cfg.curriculum_transition_steps))
        if self.common_step_counter <= warmup_steps:
            velocity_factor = 0.0
        else:
            velocity_factor = min(1.0, float(self.common_step_counter - warmup_steps) / float(transition_steps))
        stepping_factor = max(0.5, 1.0 - 0.5 * velocity_factor)
        return stepping_factor, velocity_factor


    def _get_command_curriculum_scales(self, velocity_factor: float) -> tuple[float, float]:
        if velocity_factor <= 0.0:
            return 0.0, 0.0

        xy_progress = max(0.0, min(1.0, velocity_factor / max(self.cfg.yaw_tracking_start_ratio, 1.0e-6)))
        xy_scale = self.cfg.xy_command_min_curriculum_scale + (
            1.0 - self.cfg.xy_command_min_curriculum_scale
        ) * (xy_progress ** self.cfg.xy_command_curriculum_power)
        yaw_scale = max(
            0.0,
            min(
                1.0,
                (velocity_factor - self.cfg.yaw_tracking_start_ratio)
                / max(1.0 - self.cfg.yaw_tracking_start_ratio, 1.0e-6),
            ),
        )
        return xy_scale, yaw_scale


    def _get_reward_factor(self, reward_name: str, stepping_factor: float, velocity_factor: float) -> float:
        factor_kind = self._get_reward_factor_kind(reward_name)
        if factor_kind in {"velocity", "yaw_velocity"}:
            if factor_kind == "yaw_velocity" and velocity_factor < self.cfg.yaw_tracking_start_ratio:
                return 0.0
            return velocity_factor
        if factor_kind == "stepping":
            return stepping_factor
        return 1.0


    def _get_reward_factor_kind(self, reward_name: str) -> str:
        if reward_name == "track_ang_vel_z_exp":
            return "yaw_velocity"
        if reward_name in self._VELOCITY_REWARD_NAMES:
            return "velocity"
        if reward_name in self._STEPPING_REWARD_NAMES:
            return "stepping"
        return "constant"


    def _sample_uniform(self, value_range: tuple[float, float], batch_size: int) -> torch.Tensor:
        low, high = value_range
        return low + (high - low) * torch.rand(batch_size, device=self.device)


    def enable_manual_command_override(self, enabled: bool):
        self._manual_command_override_enabled = bool(enabled)
        if enabled:
            self._command_time_left.zero_()
            self._commands.copy_(self._manual_commands)
            self._update_command_buffers()


    def set_manual_commands(self, commands: torch.Tensor | list[float] | tuple[float, float, float]):
        command_tensor = torch.as_tensor(commands, dtype=torch.float32, device=self.device)
        if command_tensor.ndim == 1:
            command_tensor = command_tensor.unsqueeze(0).repeat(self.num_envs, 1)
        elif command_tensor.shape[0] == 1 and self.num_envs > 1:
            command_tensor = command_tensor.repeat(self.num_envs, 1)
        elif command_tensor.shape[0] != self.num_envs:
            raise ValueError(
                f"Manual command batch size mismatch: expected 1 or {self.num_envs}, got {command_tensor.shape[0]}"
            )
        if command_tensor.shape[1] != 3:
            raise ValueError(f"Manual commands must have shape (*, 3), got {tuple(command_tensor.shape)}")
        self._manual_commands.copy_(command_tensor)
        if self._manual_command_override_enabled:
            self._commands.copy_(self._manual_commands)
            self._update_command_buffers()
