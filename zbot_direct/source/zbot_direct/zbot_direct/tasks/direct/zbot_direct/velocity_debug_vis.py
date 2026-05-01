from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG


class ZbotVelocityDebugVis:
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self._goal_vel_visualizer is None:
                goal_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CommandVelocityGoal")
                goal_cfg.markers["arrow"].scale = (1.0, 0.18, 0.18)
                self._goal_vel_visualizer = VisualizationMarkers(goal_cfg)

                current_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CommandVelocityCurrent")
                current_cfg.markers["arrow"].scale = (1.0, 0.18, 0.18)
                self._current_vel_visualizer = VisualizationMarkers(current_cfg)

                goal_yaw_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CommandYawGoal")
                goal_yaw_cfg.markers["arrow"].scale = (0.7, 0.16, 0.16)
                self._goal_yaw_visualizer = VisualizationMarkers(goal_yaw_cfg)

                current_yaw_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CommandYawCurrent")
                current_yaw_cfg.markers["arrow"].scale = (0.7, 0.16, 0.16)
                self._current_yaw_visualizer = VisualizationMarkers(current_yaw_cfg)

            self._goal_vel_visualizer.set_visibility(True)
            self._current_vel_visualizer.set_visibility(True)
            self._goal_yaw_visualizer.set_visibility(True)
            self._current_yaw_visualizer.set_visibility(True)
        else:
            if self._goal_vel_visualizer is not None:
                self._goal_vel_visualizer.set_visibility(False)
                self._current_vel_visualizer.set_visibility(False)
                self._goal_yaw_visualizer.set_visibility(False)
                self._current_yaw_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        if (
            self._goal_vel_visualizer is None
            or self._current_vel_visualizer is None
            or self._goal_yaw_visualizer is None
            or self._current_yaw_visualizer is None
        ):
            return

        self._update_state_buffers()
        base_pos_w = self.base_pos_w.clone()
        base_pos_w[:, 2] += 0.28
        goal_scales, goal_orientations = self._resolve_xy_velocity_to_arrow(
            self._commands[:, :2], self._goal_vel_visualizer
        )
        actual_planar_vel = torch.cat((self.base_lin_vel_forward_b, self.base_lin_vel_side_b), dim=1)
        current_scales, current_orientations = self._resolve_xy_velocity_to_arrow(
            actual_planar_vel, self._current_vel_visualizer
        )
        yaw_goal_pos, yaw_goal_scales, yaw_goal_orientations = self._resolve_yaw_rate_to_marker(self._commands[:, 2])
        yaw_current_pos, yaw_current_scales, yaw_current_orientations = self._resolve_yaw_rate_to_marker(
            self.base_ang_vel_z_b.squeeze(-1)
        )
        self._goal_vel_visualizer.visualize(base_pos_w, goal_orientations, goal_scales)
        self._current_vel_visualizer.visualize(base_pos_w, current_orientations, current_scales)
        self._goal_yaw_visualizer.visualize(yaw_goal_pos, yaw_goal_orientations, yaw_goal_scales)
        self._current_yaw_visualizer.visualize(yaw_current_pos, yaw_current_orientations, yaw_current_scales)


    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, visualizer: VisualizationMarkers
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= 0.15 + torch.linalg.norm(xy_velocity, dim=1) * 2.2
        world_xy_direction = (
            xy_velocity[:, 0:1] * self.base_dir_forward_w[:, :2]
            + xy_velocity[:, 1:2] * self.base_dir_side_w[:, :2]
        )
        heading_angle = torch.atan2(world_xy_direction[:, 1], world_xy_direction[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat


    def _resolve_yaw_rate_to_marker(self, yaw_rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_pos_w = self.base_pos_w.clone()
        marker_pos_w = base_pos_w.clone()
        marker_pos_w[:, 2] += 0.48

        magnitude = torch.abs(yaw_rate)
        marker_scale = torch.ones(self.num_envs, 3, device=self.device)
        marker_scale[:, 0] = 0.18 + magnitude * 0.8
        marker_scale[:, 1] = 0.16
        marker_scale[:, 2] = 0.16

        roll = torch.zeros_like(yaw_rate)
        pitch = torch.where(yaw_rate >= 0.0, -0.5 * torch.pi, 0.5 * torch.pi * torch.ones_like(yaw_rate))
        yaw = torch.zeros_like(yaw_rate)
        marker_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        return marker_pos_w, marker_scale, marker_quat


