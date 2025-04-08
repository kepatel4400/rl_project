#!/bin/bash

task_name="v0_obj_rand_pos_after_robot_pose_fix"

# Check if the load model argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_model>"
  exit 1
fi

# Run the training script with the provided model path
python scripts/stage_6_train_agent_v2.py --task-name "${task_name}" --run-episode --load-model "$1" 2>&1 | tee -a "outputs/$(date +'%Y%m%d_%H%M%S')_${task_name}_run_log.txt"