#!/usr/bin/env bash
set -euo pipefail

ISAACLAB="${ISAACLAB:-$HOME/isaaclab}"

"$ISAACLAB/isaaclab.sh" -p -m pip uninstall -y zbot_direct || true
"$ISAACLAB/isaaclab.sh" -p -m pip install -e zbot_direct/source/zbot_direct/
