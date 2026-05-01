# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained velocity policy with keyboard-controlled vx/vy/wz commands."""

import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from checkpoint_compat import load_runner_checkpoint_compat  # isort: skip


parser = argparse.ArgumentParser(description="Play a trained velocity policy with keyboard commands.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

_SOURCE_PATH = Path(__file__).resolve().parents[2] / "source" / "zbot_direct"
sys.path.insert(0, str(_SOURCE_PATH))
if "zbot_direct" in sys.modules and getattr(sys.modules["zbot_direct"], "__file__", None) is None:
    del sys.modules["zbot_direct"]

from zbot_direct import register_tasks

register_tasks()

installed_version = metadata.version("rsl-rl-lib")


def _checkpoint_from_dir(directory: str) -> str:
    checkpoints = [
        filename for filename in os.listdir(directory) if filename.endswith(".pt") or filename.endswith(".pth")
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoint found in directory: '{directory}'")
    return os.path.join(directory, sorted(checkpoints)[-1])


def _create_keyboard(env) -> Se2Keyboard:
    command_ranges = env.unwrapped.cfg.command_ranges
    return Se2Keyboard(
        Se2KeyboardCfg(
            sim_device=env.unwrapped.device,
            v_x_sensitivity=float(max(abs(v) for v in command_ranges["lin_vel_x"])),
            v_y_sensitivity=float(max(abs(v) for v in command_ranges["lin_vel_y"])),
            omega_z_sensitivity=float(max(abs(v) for v in command_ranges["ang_vel_z"])),
        )
    )


def _enable_manual_velocity_control(env) -> None:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "common_step_counter") and hasattr(unwrapped, "cfg"):
        warmup_steps = getattr(unwrapped.cfg, "curriculum_warmup_steps", 0)
        transition_steps = getattr(unwrapped.cfg, "curriculum_transition_steps", 0)
        unwrapped.common_step_counter = int(warmup_steps + transition_steps)
    if hasattr(unwrapped, "enable_manual_command_override"):
        unwrapped.enable_manual_command_override(True)
        unwrapped.set_manual_commands(torch.zeros(3, device=unwrapped.device))
    if hasattr(unwrapped, "set_debug_vis") and not args_cli.headless:
        unwrapped.set_debug_vis(True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif agent_cfg.load_run and os.path.isfile(agent_cfg.load_run):
        resume_path = os.path.abspath(agent_cfg.load_run)
    elif agent_cfg.load_run and os.path.isdir(agent_cfg.load_run):
        resume_path = _checkpoint_from_dir(agent_cfg.load_run)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    _enable_manual_velocity_control(raw_env)

    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    load_runner_checkpoint_compat(runner, resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    keyboard = _create_keyboard(env)
    print(keyboard)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    while simulation_app.is_running():
        start_time = time.time()
        env.unwrapped.set_manual_commands(keyboard.advance())
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
