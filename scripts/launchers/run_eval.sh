#!/bin/bash
set -eo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

task_name="v0_obj_rand_pos_after_robot_pose_fix"

# Require explicit trust before passing the model to Python.
if [[ $# -ne 2 || -z "$1" || "$2" != --trust-checkpoint ]]; then
  echo "Usage: $0 <path_to_model> --trust-checkpoint" >&2
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p outputs/eval

# Get timestamp for the experiment
timestamp=$(date +'%Y%m%d_%H%M%S')

# Create experiment directory
exp_dir="outputs/eval/${timestamp}_${task_name}"
mkdir -p "${exp_dir}"

# Run the evaluation
python "$repo_root/scripts/stages/stage_6_train_agent_v2.py" --task-name "${task_name}" --eval-only --load-model="$1" --trust-checkpoint 2>&1 | tee -a "${exp_dir}/eval_log.txt"

echo "Evaluation complete for task: ${task_name}"
echo "Results saved to: ${exp_dir}"
