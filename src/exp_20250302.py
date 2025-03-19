import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.geometry import calcurate_concept_matrics_rank
from src.utils.model_analysis import (
    compute_inner_product_LOO,
    get_concept_vector,
    get_hidden_layer_n,
)
from src.utils.preprocess_data import get_counterfactual_pairs
from src.utils.visualization import show_histogram_LOO
from src.utils.dimension_reduction.ica import apply_ica
from src.utils.dimension_reduction.pca import apply_pca


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
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4", help='Comma-separated list of target layers')
    parser.add_argument("--num_sample", type=int, default=1000)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--concept_vectorize_strategy", type=str, default="embedding"
    )  # embedding or unembedding or unembeddin2embedding
    parser.add_argument("--embedding_batch_size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    dataset_type = args.dataset_type
    concept_direction_type = args.concept_direction_type
    norm_type = args.norm_type
    prompt_type = args.prompt_type
    embedding_strategy = args.embedding_strategy
    device_id = args.device_id
    concept_vectorize_strategy = args.concept_vectorize_strategy
    embedding_batch_size = args.embedding_batch_size
    target_layers = [int(layer) for layer in args.target_layers.split(',')]

    # Basic config
    model_name = model_path.split("/")[1].lower()
    num_sample = args.num_sample

    # Load model
    device = torch.device(f"cuda:{device_id}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    num_hidden_layers: int = model.config.num_hidden_layers
    unembedding = model.lm_head.weight.detach()

    values_list_str: list[str] = [line.strip() for line in open("/home/itai/research/PersonalValuesGeometry/datasets/values.txt")]

    for target_layer in target_layers:
        ## results path
        SAVE_PATH_STRUCTURE = f"{model_name}/embeddings/layer{target_layer}/{dataset_type}/{concept_direction_type}/{norm_type}/{prompt_type}"
        # counterfactual_output_path = f"/home/itai/research/PersonalValuesGeometry/datasets/ValueNet/schwartz/{concept_direction_type}/{norm_type}"
        analyzed_figure_path = (
            f"/home/itai/research/PersonalValuesGeometry/figures/{SAVE_PATH_STRUCTURE}"
        )
        # generation_random_output_path = f"/home/itai/research/PersonalValuesGeometry/generated/{SAVE_PATH_STRUCTURE}/random.json"
        # generation_counterfactual_output_path = f"/home/itai/research/PersonalValuesGeometry/generated/{SAVE_PATH_STRUCTURE}/counterfactual.json"

        random_txt_path = f"/home/itai/research/PersonalValuesGeometry/datasets/ValueNet/schwartz/random_pairs/{norm_type}/random_1000_pairs.txt"

        # inner_product_matrix_path = (
        #     f"/home/itai/research/PersonalValuesGeometry/matrices/{SAVE_PATH_STRUCTURE}"
        # )

        print("\n*===== Args =====*")
        print(f"{model_path=}")
        print(f"{dataset_type=}")
        print(f"{concept_direction_type=}")
        print(f"{norm_type=}")
        print(f"{prompt_type=}")
        print(f"{embedding_strategy=}")
        print(f"{target_layer=}")
        print(f"{num_sample=}")
        print(f"!{concept_vectorize_strategy=}")
        print("*===============*\n")

        # Random Pair
        print("[Random Pair] random文書pairを取得 ...")
        # random_pairs = get_sequence_pairs(random_txt_path, int(num_sample))
        random_positive_sequences, random_negative_sequences = get_counterfactual_pairs(
            random_txt_path, prompt_type=prompt_type, num_sample=int(num_sample)
        )

        print(f"{random_positive_sequences[0]=}")

        print("[Random Pair] positive文章のembeddingを計算 ...")
        random_positive_embeddings = get_hidden_layer_n(
            model=model,
            tokenizer=tokenizer,
            sequences=random_positive_sequences,
            n_layer=target_layer,
            embedding_strategy=embedding_strategy,
            batch_size=embedding_batch_size,
        )

        print("[Random Pair] positive文章のembeddingをPCAで次元削減 ...")
        reduced_pca_random_positive_embeddings = apply_ica(random_positive_embeddings, n_components=10)
        print("[Random Pair] positive文章のembeddingをICAで次元削減 ...")
        reduced_ica_random_positive_embeddings = apply_pca(random_positive_embeddings, n_components=10)

        print("[Random Pair] negative文章のembeddingを計算 ...")
        random_negative_embeddings = get_hidden_layer_n(
            model,
            tokenizer,
            random_negative_sequences,
            target_layer,
            embedding_strategy=embedding_strategy,
            batch_size=embedding_batch_size,
        )
        recuded


        print("[Random Pair] positive/negativeのembedding差分を計算 ...")
        random_diff_embeddings = random_positive_embeddings - random_negative_embeddings

        print("[Random Pair] 内積をLOOで計算 ...")
        random_inner_product_LOO = compute_inner_product_LOO(
            diff_embeddings=random_diff_embeddings
        )

        print("[Random Pair] 差分ベクトルの行列に対して行列のrankを計算 ...")
        matrics_rank_save_path = f"/home/itai/research/PersonalValuesGeometry/experiments_results/matrix_rank/{SAVE_PATH_STRUCTURE}"
        os.makedirs(matrics_rank_save_path, exist_ok=True)
        with open(f"{matrics_rank_save_path}/random.csv", "w") as f:
            for epsilon in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                rank = calcurate_concept_matrics_rank(concept_matrics=random_diff_embeddings, epsilon=epsilon)
                f.write(f"{epsilon},{rank}\n")

        print("[Random Pair] 概念ベクトルを計算 ...")
        random_concept_vector = get_concept_vector(diff_embeddings=random_diff_embeddings)

        # Counterfactual Pair
        all_values_inner_product_LOO = []
        for value_str in values_list_str:
            print("=" * 20 + f"Value: {value_str}" + "=" * 20)
            print("[Counterfactual Pair] counterfactual文書pairを取得 ...")
            counter_factual_data_path = f"/home/itai/research/PersonalValuesGeometry/datasets/ValueNet/schwartz/{concept_direction_type}/{norm_type}/[{value_str}(Positive) - {value_str}(Negative)].txt"
            concept_positive_sequences, concept_negative_sequences = (
                get_counterfactual_pairs(
                    counter_factual_data_path,
                    prompt_type=prompt_type,
                    num_sample=int(num_sample),
                )
            )

            print(f"[Concept {value_str} Pair] positive文章のembeddingを計算 ...")
            concpet_positive_embeddings = get_hidden_layer_n(
                model=model,
                tokenizer=tokenizer,
                sequences=concept_positive_sequences,
                n_layer=target_layer,
                embedding_strategy=embedding_strategy,
                batch_size=embedding_batch_size,
            )

            print(f"[Concept {value_str} Pair] negative文章のembeddingを計算 ...")
            concept_negative_embedings = get_hidden_layer_n(
                model=model,
                tokenizer=tokenizer,
                sequences=concept_negative_sequences,
                n_layer=target_layer,
                embedding_strategy=embedding_strategy,
                batch_size=embedding_batch_size,
            )

            print(f"[Concept {value_str} Pair] positive/negativeのembedding差分を計算 ...")
            concept_diff_embeddings = concpet_positive_embeddings - concept_negative_embedings

            print(f"[Concept {value_str} Pair] 内積をLOOで計算 ...")
            concept_inner_product_LOO = compute_inner_product_LOO(
                diff_embeddings=concept_diff_embeddings
            )
            all_values_inner_product_LOO.append(concept_inner_product_LOO)

            print(f"[Concept {value_str} Pair] 差分ベクトルの行列に対して行列のrankを計算 ...")
            with open(f"{matrics_rank_save_path}/{value_str}.csv", "w") as f:
                for epsilon in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                    rank = calcurate_concept_matrics_rank(concept_matrics=concept_diff_embeddings, epsilon=epsilon)
                    f.write(f"{epsilon},{rank}\n")

            print(f"[Concept {value_str} Pair] 概念ベクトルを計算 ...")
            value_concept_vector = get_concept_vector(
                diff_embeddings=concept_diff_embeddings
            )

        # Visualize LOO histograms
        show_histogram_LOO(
            all_inner_product_LOO=all_values_inner_product_LOO,
            random_inner_product_LOO=random_inner_product_LOO,
            concept_names=values_list_str,
            save_dir=analyzed_figure_path,
            cols=4,
            title_fontsize=12,
            is_pca=False,
        )
