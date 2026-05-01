#!/usr/bin/env bash
set -euo pipefail

ISAACLAB="${ISAACLAB:-$HOME/isaaclab}"

"$ISAACLAB/isaaclab.sh" -p zbot_direct/scripts/list_envs.py
