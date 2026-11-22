#!/bin/bash
set -eo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -ne 1 || "$1" != --trust-checkpoint ]]; then
  echo "Usage: $0 --trust-checkpoint" >&2
  exit 1
fi

# Define arrays for experiment configurations with one-to-one mapping
exp_names=("exp_03" "exp_04" "exp_05" "exp_06")
observation_types=("vision_only" "proprio_only" "multimodal" "multimodal_no_goal")
model_names=(
  "final_model_sac_20250326_031046"
  "final_model_sac_20250326_175631" 
  "interrupted_model_sac_20250327_083255"
  "interrupted_model_sac_20250327_141302"
)
difficulty_levels=(0 1 2 3 4)

# Track total runs
total_runs=0

# Print configuration summary
echo "======= EXPERIMENT SWEEP CONFIGURATION ======="
echo "Total configurations: ${#exp_names[@]} experiment types × ${#difficulty_levels[@]} difficulty levels = $((${#exp_names[@]} * ${#difficulty_levels[@]})) runs"
echo ""
echo "Experiment mappings:"
for i in {0..3}; do
  echo "  ${exp_names[$i]} → ${observation_types[$i]} → ${model_names[$i]}"
done
echo "=============================================="
echo ""

# Loop through each experiment configuration
for i in {0..3}; do
  exp_name=${exp_names[$i]}
  observation_type=${observation_types[$i]}
  model_name=${model_names[$i]}
  
  echo "Processing experiment: ${exp_name} with ${observation_type}"
  
  # For each configuration, loop through all difficulty levels
  for difficulty_level in "${difficulty_levels[@]}"; do
    task_name="${exp_name}_${observation_type}_eval_difficulty_level_${difficulty_level}"
    
    echo "  Running task: ${task_name} [$(( total_runs + 1 ))/20]"
    
    # Run the evaluation script
    python "$repo_root/scripts/stage_6_train_agent_exp_03_eval.py" \
      --eval-only --trust-checkpoint \
      --difficulty-level ${difficulty_level} \
      --observation-type ${observation_type} \
      --load-model "$repo_root/checkpoints/models_${exp_name}_${observation_type}/${model_name}.zip" 2>&1 | \
      tee -a "eval_output_${task_name}_$(date +'%Y%m%d_%H%M%S').log"
    
    # Increment counter
    total_runs=$((total_runs + 1))
    
    echo "  Completed task: ${task_name}"
    echo ""
  done
  
  echo "Completed all difficulty levels for ${exp_name}"
  echo "=============================================="
  echo ""
done

echo "All experiments completed! Total runs: ${total_runs}"
