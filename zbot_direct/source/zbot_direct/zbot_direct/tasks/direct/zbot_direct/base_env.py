from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor


class ZbotBaseEnv(DirectRLEnv):
    """Shared scene and robot wiring for direct ZBot tasks."""

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._integrate_joint_position_actions(actions)

    def _integrate_joint_position_actions(
        self,
        actions: torch.Tensor,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
    ):
        lower_limit = -torch.pi if lower_limit is None else lower_limit
        upper_limit = torch.pi if upper_limit is None else upper_limit
        self._actions = torch.tanh(actions.clone())
        self.p_delta[:] += (
            torch.pi
            * self.joint_speed_limit
            * self.cfg.action_scale
            * self._actions
            * self.step_dt
        )
        self.p_delta = torch.clip(self.p_delta, lower_limit, upper_limit)
        self._processed_actions = self.p_delta + self._robot.data.default_joint_pos

    def _setup_body_indices(self):
        self.base_body_idx = self._robot.find_bodies("base")[0]
        self.feet_body_idx = self._robot.find_bodies("foot.*")[0]
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(
            "base|a.*|b.*"
        )

    def _setup_locomotion_buffers(self, joint_speed_limit):
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._base_id, _ = self._contact_sensor.find_bodies("base|a.*|b.*")
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("a.*|b.*")
        self.base_body_idx = self._robot.find_bodies("base")[0]
        self.feet_body_idx = self._robot.find_bodies("foot.*")[0]

        self.feet_contact_forces_last = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float
        )
        self.base_pos_w = self._robot.data.body_pos_w[:, self.feet_body_idx]
        self.feet_down_pos_last = self.base_pos_w.clone()
        self.feet_step_length = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_air_times = torch.zeros((self.num_envs, 2), device=self.device)
        self.base_heading_x_sum = torch.zeros((self.num_envs, 1), device=self.device)
        self.dead_time = torch.zeros(self.num_envs, device=self.device)
        self.feet_force_sum = torch.zeros(self.num_envs, device=self.device)
        self.robot_com = torch.zeros(self.num_envs, 3, device=self.device)

        self.joint_speed_limit = joint_speed_limit
        self.p_delta = torch.zeros_like(self._robot.data.default_joint_pos)
        self.loop_count = 0
        self._init_reward_functions(self.cfg)

    def _setup_reward_functions(self):
        self._init_reward_functions(self.cfg)

    def _init_reward_functions(self, cfg) -> None:
        if hasattr(cfg, "reward_cfg"):
            reward_scales = cfg.reward_cfg["reward_scales"]
        else:
            reward_scales = cfg.reward_scales
        self.reward_scales = dict(reward_scales)
        self.reward_functions = {}
        self._episode_sums = {}
        for name, scale in self.reward_scales.items():
            self.reward_scales[name] = scale * self.step_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self._episode_sums[name] = self._new_env_float_buffer()

    def _new_env_float_buffer(self):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _write_robot_state(self, env_ids, joint_pos, joint_vel):
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self._episode_sums[name] += rew
        terminated_ids = self.reset_terminated.nonzero(as_tuple=False).squeeze(-1)
        reward[terminated_ids] -= 20.0
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._update_contact_state()
        died = self._combine_termination_terms(self._termination_terms())
        return died, time_out

    def _update_contact_state(self) -> None:
        self.net_contact_forces = self._contact_sensor.data.net_forces_w_history
        self.feet_contact_forces = torch.mean(
            self.net_contact_forces[:, :, self._feet_ids, 2],
            dim=1,
        ).squeeze()
        self.feet_air_times = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        self.feet_contact_times = self._contact_sensor.data.current_contact_time[
            :, self._feet_ids
        ]

    def _combine_termination_terms(self, terms: tuple[torch.Tensor, ...]) -> torch.Tensor:
        died = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        for term in terms:
            died |= term.squeeze()
        return died

    def _base_contact_termination(self, threshold: float) -> torch.Tensor:
        return torch.any(
            torch.max(
                torch.norm(self.net_contact_forces[:, :, self._base_id], dim=-1), dim=1
            )[0]
            > threshold,
            dim=1,
        )

    def _termination_terms(self) -> tuple[torch.Tensor, ...]:
        return (
            self._base_contact_termination(1.0),
            self.base_pos_w[:, 2] < self.cfg.termination_height,
            self.base_pos_y_err.abs() > 0.5,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
            -1.0, 1.0
        )
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._write_robot_state(env_ids, joint_pos, joint_vel)
        self.p_delta[env_ids] = 0.0
        self.feet_down_pos_last[env_ids] = (
            self._robot.data.body_pos_w[:, self.base_body_idx]
        )[env_ids]
        self.base_heading_x_sum[env_ids] = 0
        self.dead_time[env_ids] = 0
        self.feet_force_sum[env_ids] = 0

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0

        extras["Episode_Termination/body_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"] = extras
