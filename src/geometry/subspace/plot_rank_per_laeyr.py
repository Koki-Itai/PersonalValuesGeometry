import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    parser.add_argument("--dataset_type", type=str, default="valuenet")
    parser.add_argument("--concept_direction_type", type=str, default="pos2neg")
    parser.add_argument("--norm_type", type=str, default="base")
    parser.add_argument("--prompt_type", type=str, default="reflection")
    parser.add_argument("--embedding_strategy", type=str, default="last")
    parser.add_argument(
        "--target_layers",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of target layers",
    )
    return parser.parse_args()


args = parse_args()
model_path = args.model_path
dataset_type = args.dataset_type
concept_direction_type = args.concept_direction_type
norm_type = args.norm_type
prompt_type = args.prompt_type
embedding_strategy = args.embedding_strategy
target_layers = [int(layer) for layer in args.target_layers.split(",")]
model_name = model_path.split("/")[1].lower()

matrix_rank_results_dir = (
    "/home/itai/research/PersonalValuesGeometry/experiments_results/matrix_rank"
)
values_list_str: list[str] = [
    line.strip()
    for line in open("/home/itai/research/PersonalValuesGeometry/datasets/values.txt")
]
values_list_str.append("random")

epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

rank_per_value = defaultdict(list)
for value in values_list_str:
    rank_per_epsilon = defaultdict(list)
    for target_layer in target_layers:
        save_path_struicture = f"{model_name}/embeddings/layer{target_layer}/{dataset_type}/{concept_direction_type}/{norm_type}/{prompt_type}"
        concept_rank_results_path = (
            f"{matrix_rank_results_dir}/{save_path_struicture}/{value}.csv"
        )
        tmp_df = pd.read_csv(concept_rank_results_path, header=None)
        for i in range(len(tmp_df)):
            rank = tmp_df.iloc[[i], [1]].values[0][0]
            rank_per_epsilon[i].append(rank)
        rank_per_value[value] = rank_per_epsilon

for i, epsilon in enumerate(epsilons):
    print(f"epsilon: {epsilon}")
    plt.figure()
    plt.title(f"model: {model_name}\nepsilon: {epsilon}\nprompt_type: {prompt_type}")
    for value in values_list_str:
        rank_dist_per_layer = rank_per_value[value][i]
        plt.plot(target_layers, rank_dist_per_layer, label=f"{value}")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Layer")
        plt.ylabel("Rank")
    plt.show()
    save_path = f"{matrix_rank_results_dir}/plots/{prompt_type}/{model_name}_embeddings_{dataset_type}_{concept_direction_type}_{norm_type}_epsilon_{epsilon}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
