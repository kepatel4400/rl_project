#!/bin/bash

task_name="v0_obj_rand_pos_after_robot_pose_fix"

# Check if the load model argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_model>"
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
python scripts/stage_6_train_agent_v2.py --task-name "${task_name}" --eval-only --load-model "$1" 2>&1 | tee -a "${exp_dir}/eval_log.txt"

echo "Evaluation complete for task: ${task_name}"
echo "Results saved to: ${exp_dir}"