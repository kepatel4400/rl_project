#!/bin/bash
set -eo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# This script runs the robotic pushing agent training with a YAML configuration
# and logs all output to a timestamped file in the experiment directory

# Define the task name and config file
task_name="only_test_del"
config_yaml="$repo_root/config/train/set_1_simple_reward_obj_rand_pose.yaml"

# Create output directory if it doesn't exist
mkdir -p outputs/train

# Get timestamp for the experiment
timestamp=$(date +'%Y%m%d_%H%M%S')

# Create experiment directory
exp_dir="outputs/train/${timestamp}_${task_name}"
mkdir -p "${exp_dir}"

# Run the training script with the YAML config
# Output will be captured in the experiment directory
python "$repo_root/scripts/stages/stage_6_train_agent_v3.py" --config "${config_yaml}" --task-name "${task_name}" 2>&1 | tee -a "${exp_dir}/training_log.txt"

echo "Training complete for task: ${task_name}"
echo "Results saved to: ${exp_dir}"