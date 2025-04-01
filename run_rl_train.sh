#!/bin/bash

# Define the task name
task_name="exp_11_multimodal_no_goal_vary_target"
observation_type="multimodal_no_goal" # vision_only, proprio_only, multimodal, multimodal_no_goal

# Run the training script
python scripts/stage_6_train_agent_exp_03_ver2.py --total-timesteps 3000000 --learning-rate 3e-4 --buffer-size 3000000 --batch-size 1024 --n-envs 20 --parallel --max-steps 500 --train-freq 5 --gradient-steps 10 --learning-starts 20000 --tensorboard --checkpoint-freq 25000 --eval-freq 25000 --observation-type ${observation_type} --model-dir "models_${task_name}" 2>&1 | tee -a "output_${task_name}_$(date +'%Y%m%d_%H%M%S').log"
