import isaaclab.sim as sim_utils

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


DEFAULT_VELOCITY_REWARD_SCALES = {
    "track_lin_vel_xy_exp": 8.0,
    "track_ang_vel_z_exp": 0.5,
    "command_forward_progress": 2.0,
    "command_speed_shortfall": -1.0,
    "lateral_velocity_match": 0.5,
    "command_stillness": -0.2,
    "lin_vel_xy_variance_l2": -0.05,
    "ang_vel_z_variance_l2": 0.0,
    "leg_toggle": 5.0,
    "single_leg_support": 0.8,
    "airtime_balance": -1.0,
    "double_support": -0.4,
    "action_rate_l2": -0.02,
    "dof_torques_l2": -2.0e-4,
    "joint_deviation_l1": -0.05,
    "feet_slide": -0.5,
}


@configclass
class ZbotVelocityEnvCfg(DirectRLEnvCfg):
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.0,
        track_air_time=True,
        track_pose=True,
    )

    episode_length_s = 20.0
    decimation = 2
    action_scale = 1.0
    state_space = 0

    command_resample_time_range = (4.0, 8.0)
    standing_probability = 0.1
    command_ranges = {
        "lin_vel_x": (-0.8, 0.8),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-1.0, 1.0),
    }
    stepping_frequency_range = (0.8, 1.6)
    stepping_clearance_target = 0.10
    curriculum_warmup_steps = 2000
    curriculum_transition_steps = 8000
    yaw_tracking_start_ratio = 0.75
    xy_command_min_curriculum_scale = 0.25
    xy_command_curriculum_power = 1.5

    termination_height = 0.05
    base_contact_force_threshold = 1000.0
    termination_gravity_xy = 0.7
    non_foot_contact_force_threshold = 1.0
    min_feet_separation = 0.10

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    reward_scales = DEFAULT_VELOCITY_REWARD_SCALES
