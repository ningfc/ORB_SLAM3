#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <path_to_vocabulary> <path_to_settings> [trajectory_file_name]" >&2
  echo "Example: $0 ./Vocabulary/ORBvoc.txt ./Examples/Stereo-Inertial/Config.yaml" >&2
  exit 1
fi

VOCAB_PATH="$1"
SETTINGS_PATH="$2"
TRAJ_PATH="${3:-}"

BIN="$ROOT_DIR/Examples/Stereo-Inertial/stereo_inertial_cyperstereo"
if [[ ! -x "$BIN" ]]; then
  echo "Executable not found: $BIN" >&2
  echo "Please build ORB_SLAM3 first." >&2
  exit 1
fi

if [[ -n "$TRAJ_PATH" ]]; then
  "$BIN" "$VOCAB_PATH" "$SETTINGS_PATH" "$TRAJ_PATH"
else
  "$BIN" "$VOCAB_PATH" "$SETTINGS_PATH"
fi
