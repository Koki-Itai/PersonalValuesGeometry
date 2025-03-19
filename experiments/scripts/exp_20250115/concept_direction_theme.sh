bash /home/itai/research/linear_rep_geometry/setup.sh

MODEL_PATHS=("meta-llama/Llama-3.2-3B-Instruct")
DATASET_TYPES=("valuenet")
CONCEPT_DIRECTIONS=("pos2neg")
NORM_TYPES=("base")
PROMPT_TYPES=("theme")
N_LAYERS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28")

for model_path in "${MODEL_PATHS[@]}"; do
    for dataset_type in "${DATASET_TYPES[@]}"; do
        for concept_direction_type in "${CONCEPT_DIRECTIONS[@]}"; do
            for norm_type in "${NORM_TYPES[@]}"; do
                for prompt_type in "${PROMPT_TYPES[@]}"; do
                    for n_layer in "${N_LAYERS[@]}"; do
                        python /home/itai/research/PersonalValuesGeometry/src/concept_ica_vec_embedding.py \
                            --model_path $model_path \
                            --dataset_type $dataset_type \
                            --concept_direction_type $concept_direction_type \
                            --norm_type $norm_type \
                            --prompt_type $prompt_type \
                            --embedding_strategy "last" \
                            --target_layer $n_layer \
                            --num_sample 5000 \
                            --device_id 2
                    done
                done
            done
        done
    done
done

echo "Done"