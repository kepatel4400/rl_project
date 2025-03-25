#!/bin/bash

# Define the task name
task_name="exp_02_curriculum_learning_ent_coef_0p1_relaxed_reward"

# Run the training script
python scripts/stage_6_train_agent_exp_02.py --total-timesteps 10000000 --learning-rate 3e-4 --buffer-size 3000000 --batch-size 1024 --n-envs 20 --parallel --max-steps 500 --train-freq 5 --gradient-steps 10 --learning-starts 20000 --tensorboard --checkpoint-freq 25000 --eval-freq 25000 --model-dir "models_${task_name}" 2>&1 | tee -a "output_${task_name}_$(date +'%Y%m%d_%H%M%S').log"
