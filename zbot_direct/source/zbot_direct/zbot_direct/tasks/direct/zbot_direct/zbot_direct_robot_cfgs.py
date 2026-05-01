from __future__ import annotations

import numpy as np

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


USD_DIR = "assets/zbot_usd/zbot/"

ZBOT_2S_USD = "zbot_2s_v0.usd"
ZBOT_6S_USD = "zbot_6s_new.usd"
ZBOT_6W_USD = "zbot_6w_v1.usd"
ZBOT_8S_V0_USD = "zbot_8s_v0.usd"
ZBOT_8S_HUMAN_USD = "zbot_8s_human.usd"
ZBOT_8S_HUMAN_2_USD = "zbot_8s_human_2.usd"
ZBOT_8S_NEW_USD = "zbot_8s_new.usd"
ZBOT_8S_SNAKE_V0_USD = "zbot_8s_snake_v0.usd"


def _deg(value: float) -> float:
    return value / 180.0 * np.pi


def _spawn_cfg(usd_file: str, *, contact_sensors: bool = True) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=USD_DIR + usd_file,
        activate_contact_sensors=contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    )


def _joint_vel(num_dof: int) -> dict[str, float]:
    return {f"joint{i}": 0.0 for i in range(num_dof)}


def _zbot_8s_cfg(
    usd_file: str,
    *,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    joint_pos: dict[str, float],
) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=_spawn_cfg(usd_file),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos=joint_pos,
            joint_vel=_joint_vel(8),
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "zbot_eight": ImplicitActuatorCfg(
                joint_names_expr=["joint.*"],
                effort_limit=2000,
                velocity_limit=1000,
                stiffness=50.0,
                damping=5.0,
                friction=0.0,
            ),
        },
    )


ZBOT_D_6S_CFG = ArticulationCfg(
    spawn=_spawn_cfg(ZBOT_6S_USD),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.06, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.312,
            "joint2": 0.837,
            "joint3": -2.02,
            "joint4": 2.02,
            "joint5": -0.837,
            "joint6": -0.312,
        },
        joint_vel={"joint[1-6]": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "zbot_six": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=2000,
            velocity_limit=1000,
            stiffness=50.0,
            damping=5.0,
            friction=0.0,
        ),
    },
)

ZBOT_D_8S_V0_CFG = ArticulationCfg(
    spawn=_spawn_cfg(ZBOT_8S_V0_USD),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        rot=(0.707, 0.0, 0.707, 0.0),
        joint_pos={"joint[1-8]": 0.0},
        joint_vel={"joint[1-8]": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "zbot_eight": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=20,
            velocity_limit=10,
            stiffness=20,
            damping=0.5,
            friction=0.0,
        ),
    },
)

ZBOT_D_8S_HUMAN_NEUTRAL_CFG = _zbot_8s_cfg(
    ZBOT_8S_HUMAN_USD,
    pos=(0.0, -0.06, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "joint0": 0,
        "joint1": 0.312,
        "joint2": 0.837,
        "joint3": -2.02,
        "joint4": 2.02,
        "joint5": -0.837,
        "joint6": -0.312,
        "joint7": 0,
    },
)

ZBOT_D_8S_HUMAN_BIPEDAL_1_CFG = _zbot_8s_cfg(
    ZBOT_8S_HUMAN_USD,
    pos=(0.0, -0.06, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "joint0": -0.08,
        "joint1": 0.0,
        "joint2": _deg(25.0),
        "joint3": _deg(-125.0),
        "joint4": _deg(125.0),
        "joint5": _deg(-25.0),
        "joint6": -0.0,
        "joint7": 0.08,
    },
)

ZBOT_D_8S_HUMAN_2_BIPEDAL_2_CFG = _zbot_8s_cfg(
    ZBOT_8S_HUMAN_2_USD,
    pos=(0.0, -0.06, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "joint0": 0.0,
        "joint1": _deg(10.0),
        "joint2": _deg(-10.0),
        "joint3": _deg(140.0),
        "joint4": _deg(-140),
        "joint5": _deg(10.0),
        "joint6": _deg(-10.0),
        "joint7": 0.0,
    },
)

ZBOT_D_8S_HUMAN_2_BIPEDAL_3_CFG = _zbot_8s_cfg(
    ZBOT_8S_HUMAN_2_USD,
    pos=(0.0, -0.06, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "joint0": _deg(15.0),
        "joint1": _deg(10.0),
        "joint2": _deg(0.0),
        "joint3": _deg(120.0),
        "joint4": _deg(-120),
        "joint5": _deg(0.0),
        "joint6": _deg(-10.0),
        "joint7": _deg(-15.0),
    },
)

ZBOT_D_8S_HUMAN_CFG = _zbot_8s_cfg(
    ZBOT_8S_HUMAN_USD,
    pos=(0.0, -0.06, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "joint0": _deg(10.0),
        "joint1": _deg(0.0),
        "joint2": _deg(15.0),
        "joint3": _deg(-125.0),
        "joint4": _deg(125),
        "joint5": _deg(-15.0),
        "joint6": _deg(0.0),
        "joint7": _deg(-10.0),
    },
)

ZBOT_D_8S_NEW_NEUTRAL_CFG = _zbot_8s_cfg(
    ZBOT_8S_NEW_USD,
    pos=(0.0, -0.06, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "joint0": 0,
        "joint1": 0.312,
        "joint2": 0.837,
        "joint3": -2.02,
        "joint4": 2.02,
        "joint5": -0.837,
        "joint6": -0.312,
        "joint7": 0,
    },
)

ZBOT_D_8S_SNAKE_V0_CFG = _zbot_8s_cfg(
    ZBOT_8S_SNAKE_V0_USD,
    pos=(0.0, -0.0, 0.053),
    rot=(0.707, 0.0, -0.707, 0.0),
    joint_pos={
        "joint0": 0.0,
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
        "joint7": 0.0,
    },
)

ZBOT_D_8S_WHEEL_CFG = _zbot_8s_cfg(
    ZBOT_8S_NEW_USD,
    pos=(0.0, -0.0, 0.053),
    rot=(0.707, 0.707, 0.0, 0.0),
    joint_pos={
        "joint0": 1.2,
        "joint1": -1.2,
        "joint2": 1.8,
        "joint3": -1.2,
        "joint4": 1.8,
        "joint5": -1.2,
        "joint6": 1.2,
        "joint7": -1.2,
    },
)

ZBOT_D_8S_CFG = ZBOT_D_8S_V0_CFG

ZBOT_D_6W_CFG = ArticulationCfg(
    spawn=_spawn_cfg(ZBOT_6W_USD),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.785398,
            "joint3": -1.570796,
            "joint4": 1.570796,
            "joint5": -0.785398,
            "joint6": 0.0,
        },
        joint_vel={"joint[1-6]": 0.0},
    ),
    actuators={
        "zbot_six_w": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=18.0,
            velocity_limit=2.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

ZBOT_D_2S_CFG = ArticulationCfg(
    spawn=_spawn_cfg(ZBOT_2S_USD, contact_sensors=False),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 3.141593,
            "joint2": -3.141593,
        },
        joint_vel={"joint[1-2]": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "zbot_two": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=20,
            velocity_limit=10,
            stiffness=50,
            damping=0.5,
            friction=0.0,
        ),
    },
)
