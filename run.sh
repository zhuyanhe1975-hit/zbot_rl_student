#!/usr/bin/env bash
set -euo pipefail

# Activate exactly one command. Keep each command on one line so the active result is obvious.

$HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play_keyboard.py --task=Zbot-Direct-6dof-bipedal-velocity-v0 --num_envs=1 --real-time --load_run ./pth/Zbot-Direct-6dof-bipedal-velocity-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play_keyboard.py --task=Zbot-Direct-6dof-bipedal-velocity-quat-v0 --num_envs=1 --real-time --load_run ./pth/Zbot-Direct-6dof-bipedal-velocity-quat-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play_keyboard.py --task=Zbot-Direct-6dof-bipedal-velocity-imu-v0 --num_envs=1 --real-time --load_run ./pth/Zbot-Direct-6dof-bipedal-velocity-imu-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play_keyboard.py --task=Zbot-Direct-8dof-bipedal-velocity-v0 --num_envs=1 --real-time --load_run ./pth/Zbot-Direct-8dof-bipedal-velocity-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-6dof-bipedal-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-6dof-bipedal-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-6dof-bipedal-quat-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-6dof-bipedal-quat-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-6dof-bipedal-to-snake-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-6dof-bipedal-to-snake-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-6dof-bipedal-to-snake-v1 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-6dof-bipedal-to-snake-v1
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-bipedal-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-bipedal-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-bipedal-v1 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-bipedal-v1
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-bipedal-v2 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-bipedal-v2
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-bipedal-v3 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-bipedal-v3
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-bird-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-bird-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-snake-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-snake-v0
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-wheel-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-wheel-v0
