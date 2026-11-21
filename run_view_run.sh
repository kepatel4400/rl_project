#!/bin/bash
set -eo pipefail

task_name="v0_obj_rand_pos_after_robot_pose_fix"

# Require explicit trust before passing the model to Python.
if [[ $# -ne 2 || -z "$1" || "$2" != --trust-checkpoint ]]; then
  echo "Usage: $0 <path_to_model> --trust-checkpoint" >&2
  exit 1
fi

mkdir -p outputs

# Run the training script with the provided model path
python scripts/stages/stage_6_train_agent_v2.py --task-name "${task_name}" --run-episode --load-model="$1" --trust-checkpoint 2>&1 | tee -a "outputs/$(date +'%Y%m%d_%H%M%S')_${task_name}_run_log.txt"
