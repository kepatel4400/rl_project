#!/bin/bash

# Define the task name
exp_name="exp_03" # exp_03, exp_04, exp_05, exp_06
observation_type="vision_only" # vision_only, proprio_only, multimodal, multimodal_no_goal
difficulty_level=0 # 0, 1, 2, 3, 4
task_name="${exp_name}_${observation_type}_eval_difficulty_level_${difficulty_level}"
model_name="final_model_sac_20250326_031046" #final_model_sac_20250326_031046, final_model_sac_20250326_175631, interrupted_model_sac_20250327_083255, interrupted_model_sac_20250327_141302

# Run the evaluation script
python scripts/stage_6_train_agent_exp_03.py --eval-only --difficulty-level ${difficulty_level} --observation-type ${observation_type} --load-model models_${exp_name}_${observation_type}/${model_name}.zip 2>&1 | tee -a "output_${task_name}_$(date +'%Y%m%d_%H%M%S').log"