#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/itai/research/PersonalValuesGeometry

# Run setup script
bash /home/itai/research/linear_rep_geometry/setup.sh

MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
DATASET_TYPES="valuenet"
CONCEPT_DIRECTIONS="pos2neg"
NORM_TYPES="base"
PROMPT_TYPES=("explicit" "bare" "topic")

LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28)
LAYERS_STR=$(IFS=','; echo "${LAYERS[*]}")

for prompt_type in "${PROMPT_TYPES[@]}"; do
    echo "Running prompt type: $prompt_type"
    python /home/itai/research/PersonalValuesGeometry/src/geometry/subspace/plot_rank_per_laeyr.py \
        --model_path $MODEL_PATH \
        --dataset_type $DATASET_TYPES \
        --concept_direction_type $CONCEPT_DIRECTIONS \
        --norm_type $NORM_TYPES \
        --prompt_type $prompt_type \
        --target_layers $LAYERS_STR
done
