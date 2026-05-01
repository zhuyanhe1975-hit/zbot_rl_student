#!/usr/bin/env bash
set -euo pipefail

# Activate exactly one command. Keep each command on one line so the active task is obvious.

# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-velocity-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
$HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-velocity-quat-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-velocity-imu-v0 --num_envs=1024 --max_iterations=1500 --headless --load_run ./pth/Zbot-Direct-6dof-bipedal-velocity-quat-v0 --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-bipedal-velocity-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-quat-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-to-snake-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-to-snake-v1 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-bipedal-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-bipedal-v1 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-bipedal-v2 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-bipedal-v3 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-bird-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-snake-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-wheel-v0 --num_envs=1024 --max_iterations=15000 --headless --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
