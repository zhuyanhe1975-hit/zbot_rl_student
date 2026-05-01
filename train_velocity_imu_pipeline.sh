#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="/home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student"
RUN_ROOT="$LOG_ROOT/zbot_direct"
TRAIN_PY="zbot_direct/scripts/rsl_rl/train.py"

echo "[1/2] Training quaternion teacher..."
"$HOME/isaaclab/isaaclab.sh" -p "$TRAIN_PY" \
    --task=Zbot-Direct-6dof-bipedal-velocity-quat-v0 \
    --num_envs=4096 \
    --max_iterations=1500 \
    --headless \
    --log_root_path "$LOG_ROOT"

TEACHER_CHECKPOINT="$(
    find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
)"

if [[ -z "$TEACHER_CHECKPOINT" || ! -f "$TEACHER_CHECKPOINT" ]]; then
    echo "Could not find the teacher checkpoint under $RUN_ROOT" >&2
    exit 1
fi

echo "[2/2] Training IMU student from teacher checkpoint:"
echo "$TEACHER_CHECKPOINT"
"$HOME/isaaclab/isaaclab.sh" -p "$TRAIN_PY" \
    --task=Zbot-Direct-6dof-bipedal-velocity-imu-v0 \
    --num_envs=1024 \
    --max_iterations=1500 \
    --headless \
    --load_run "$TEACHER_CHECKPOINT" \
    --log_root_path "$LOG_ROOT"
