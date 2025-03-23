#!/bin/bash

# Define the task name
task_name="v0_obj_rand_pos_after_robot_pose_fix"

# Run the training script
python scripts/stage_6_train_agent.py --total-timesteps 1000000 --learning-rate 3e-4 --buffer-size 300000 --batch-size 256 --n-envs 8 --parallel --max-steps 200 --train-freq 4 --gradient-steps 4 --learning-starts 5000 --tensorboard --checkpoint-freq 25000 --eval-freq 25000 2>&1 | tee -a "output_${task_name}_$(date +'%Y%m%d_%H%M%S').log"
