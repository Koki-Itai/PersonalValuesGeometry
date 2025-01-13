import argparse
import json
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

PROMPT_TEMPLATES = {
    "reflection": "Consider the deeper meaning behind this: {}\nThis text reflects values related to:",
    "analysis": "Analyzing the underlying principles in: {}\nThe core values expressed here are:",
    "implicit": "Looking at the implicit meaning of: {}\nThe fundamental values shown are:",
    "explicit": "What values are represented in this text: {}\nThe text demonstrates values of:",
    "bare": "{}",
    "theme": "What is the main theme in this text: {}\nKey themes include:",
    "topic": "Identify the primary topics discussed in: {}\nMain topics covered are:",
}


def get_sequence_pairs(filepath: str, num_samples: int = 1000) -> list[list[str]]:
    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) > num_samples:
        lines = random.sample(lines, num_samples)
    pairs = [line.split("\t") for line in lines]

    return pairs


def generate_text(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    prompt_type: str = "bare",
    max_new_tokens: int = 100,
    batch_size: int = 16,
    max_save_step: int = 1000,
) -> list[str]:
    prompt_templates = {
        "reflection": "Consider the deeper meaning behind this: {}\nThis text reflects values related to:",
        "analysis": "Analyzing the underlying principles in: {}\nThe core values expressed here are:",
        "implicit": "Looking at the implicit meaning of: {}\nThe fundamental values shown are:",
        "explicit": "What values are represented in this text: {}\nThe text demonstrates values of:",
        "bare": "{}",
        "theme": "What is the main theme in this text: {}\nKey themes include:",
        "topic": "Identify the primary topics discussed in: {}\nMain topics covered are:",
    }

    if prompt_type not in prompt_templates:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Available types: {list(prompt_templates.keys())}"
        )

    template = prompt_templates[prompt_type]
    formatted_prompts = [template.format(prompt) for prompt in prompts]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    generated_texts = []

    # max_save_stepをバッチサイズで考慮
    effective_max_save_step = max_save_step // batch_size * batch_size

    for i in tqdm(
        range(0, len(formatted_prompts), batch_size),
        desc="Generating texts",
        total=len(formatted_prompts) // batch_size
        + bool(len(formatted_prompts) % batch_size),
    ):
        if i > effective_max_save_step:
            break
        batch_prompts = formatted_prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(
            inputs.input_ids, skip_special_tokens=True
        )

        batch_generated = [
            output[len(input_text) :].strip()
            for output, input_text in zip(decoded_outputs, decoded_inputs, strict=False)
        ]
        generated_texts.extend(batch_generated)

    return generated_texts


def save_generation_results(
    pairs: list[list[str]],
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_path: str,
    concept_name: str,
    prompt_type: str,
    max_new_tokens: int = 100,
    batch_size: int = 8,
    max_save_step: int = 4,
) -> None:
    concept_texts = [pair[0] for pair in pairs]
    non_concept_texts = [pair[1] for pair in pairs]

    generated_concept_texts = generate_text(
        model,
        tokenizer,
        concept_texts,
        prompt_type,
        max_new_tokens,
        batch_size,
        max_save_step,
    )
    generated_non_concept_texts = generate_text(
        model,
        tokenizer,
        non_concept_texts,
        prompt_type,
        max_new_tokens,
        batch_size,
        max_save_step,
    )

    results = [
        {
            "concept_text": c_text,
            "non_concept_text": nc_text,
            "generated_text_concept": gc_text,
            "generated_text_non_concept": gnc_text,
        }
        for c_text, nc_text, gc_text, gnc_text in zip(
            concept_texts,
            non_concept_texts,
            generated_concept_texts,
            generated_non_concept_texts,
            strict=False,
        )
    ]

    output_path = output_path.split(".json")[0] + f"_{concept_name}.json"
    print(f"{output_path=}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def get_hidden_layer_n(
    model,
    tokenizer,
    sequences: list[str],
    n_layer: int,
    batch_size: int = 4,
    embedding_strategy: str = "last",
):
    """
    n_layer層目のの隠れ層の行列を取得する
    Returns:
        embeddings: shape=(len(sequences), hidden_size)
    """
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, len(sequences), batch_size),
        total=total_batches,
        desc="Processing embeddings",
        unit="batch",
    )
    all_embeddings = []
    for i in pbar:
        batch_texts = sequences[i : i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=128,
        )

        device = next(model.parameters()).device
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        if embedding_strategy == "mean":
            with torch.no_grad():
                outputs = model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states
                layer_embeddings = hidden_states[
                    n_layer
                ]  # shape: (batch_size, seq_len, hidden_size)
                masked_hidden_state = layer_embeddings * attention_mask.unsqueeze(-1)
                sum_hidden_state = masked_hidden_state.sum(dim=1)
                count_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
                batch_embeddings = sum_hidden_state / count_tokens

                all_embeddings.append(batch_embeddings.cpu())

                del (
                    outputs,
                    hidden_states,
                    layer_embeddings,
                    masked_hidden_state,
                    sum_hidden_state,
                )
        elif embedding_strategy == "last":
            with torch.no_grad():
                outputs = model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states
                layer_embeddings = hidden_states[n_layer]
                last_hidden_state = layer_embeddings
                last_token_indices = attention_mask.sum(dim=1) - 1
                batch_embeddings = last_hidden_state[
                    torch.arange(last_hidden_state.size(0)), last_token_indices
                ]
                all_embeddings.append(batch_embeddings.cpu())
                del outputs, hidden_states, layer_embeddings, last_hidden_state
        else:
            raise ValueError(
                f"Invalid embedding_strategy: {embedding_strategy}. Available types: [mean, last]"
            )
    return torch.cat(all_embeddings, dim=0)


def get_counterfactual_pairs(filename: str, prompt_type: str, num_sample: int = 1000):
    """
    counterfactual pairが格納されたtxtファイルを読み込み、prompt_typeに従ってpromptを生成する
    Returns:
        base_sequences: list[str]
        target_sequences: list[str]
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Available types: {list(PROMPT_TEMPLATES.keys())}"
        )
    prompt_template = PROMPT_TEMPLATES[prompt_type]

    # counterfactual pairが格納されたtxtファイルを読み込む
    with open(filename) as f:
        lines = f.readlines()

    if len(lines) > num_sample:
        lines = random.sample(lines, num_sample)

    text_pairs = []
    for line in lines:
        if line.strip():
            base, target = line.strip().split("\t")
            prefixed_base = prompt_template.format(base) if prompt_template else base
            prefixed_target = (
                prompt_template.format(target) if prompt_template else target
            )
            text_pairs.append((prefixed_base, prefixed_target))

    base_sequences = []
    target_sequences = []

    for base, target in text_pairs:
        base_sequences.append(base)
        target_sequences.append(target)

    return base_sequences, target_sequences


def compute_inner_product_LOO(diff_embeddings, verbose=False):
    """
    Args:
        diff_embeddings: counterfactual pairの差分のembedding (shape: (num_sample, hidden_size))
    Returns:
        inner_product_LOO: Leave-One-Outで計算した内積 (shape: (num_sample, ))
    """
    products = []
    for i in range(diff_embeddings.shape[0]):
        mask = torch.ones(
            diff_embeddings.shape[0], dtype=torch.bool, device=diff_embeddings.device
        )
        mask[i] = False
        loo_diff_embeddings = diff_embeddings[mask]
        mean_loo_diff_embeddings = torch.mean(
            loo_diff_embeddings, dim=0
        )  # 対象以外の平均 (概念方向)
        loo_mean = mean_loo_diff_embeddings / torch.norm(
            mean_loo_diff_embeddings
        )  # 正規化
        product = loo_mean @ diff_embeddings[i]  # 対象の表現と概念方向の内積
        products.append(product)
        if verbose:
            print(f"{loo_diff_embeddings.shape=}")
            print(f"{mean_loo_diff_embeddings.shape=}")
            print(f"{loo_mean.shape=}")
    inner_product_LOO = torch.stack(products)
    if verbose:
        print(f"{inner_product_LOO.shape=}")
    return inner_product_LOO


def get_concept_vector(concept_embeddings, non_concept_emebeddings):
    """
    Args:
        concept_embeddings: 概念に対するembedding (shape: (num_sample, hidden_size))
        non_concept_embeddings: 概念以外のembedding (shape: (num_sample, hidden_size))
    Returns:
        concept_vector: 概念方向のベクトル (shape: (hidden_size, ))
        diff_embeddings: 概念と非概念の差分のembedding (shape: (num_sample, hidden_size))
    """
    diff_embeddings = concept_embeddings - non_concept_emebeddings
    diff_embeddings = diff_embeddings / torch.norm(diff_embeddings, dim=1, keepdim=True)
    concept_direction_vector = torch.mean(diff_embeddings, dim=0)
    return concept_direction_vector, diff_embeddings


def get_concpet_name_from_filename(filename: str):
    pattern = r"\[(.*?)\("
    match1 = re.search(pattern, filename.split("/")[-1])
    concept_name = match1.group(1)
    return concept_name


def show_histogram_LOO(
    all_inner_product_LOO: list,
    random_inner_product_LOO: torch.Tensor,
    concept_names: list,
    save_dir: str,
    cols: int = 4,
    title_fontsize: int = 12,
) -> None:
    """
    Visualize histograms comparing LOO (Leave-One-Out) analysis results between counterfactual pairs and random pairs.

    Args:
        all_inner_product_LOO: List of tensors containing LOO inner products for each concept
        random_inner_product_LOO: Tensor containing LOO inner products for random pairs
        concept_names: List of concept names corresponding to each inner product tensor
        save_dir: Directory to save the output figure
        cols: Number of columns in the subplot grid
        title_fontsize: Font size for subplot titles
        fig_name: Name for the output figure file
    """
    n_plots = len(all_inner_product_LOO)
    cols = min(cols, n_plots)
    rows = (n_plots + cols - 1) // cols

    # Create figure with appropriate size
    fig_height = 6 * rows
    fig_width = 5 * cols
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create grid specification
    gs = fig.add_gridspec(rows, cols, hspace=0.6, wspace=0.4)
    axs = []
    for i in range(rows):
        for j in range(cols):
            axs.append(fig.add_subplot(gs[i, j]))

    # Plot histograms for each concept
    for i in range(len(axs)):
        if i < n_plots:
            target = all_inner_product_LOO[i]
            baseline = random_inner_product_LOO

            # Plot histograms
            axs[i].hist(
                baseline.cpu().numpy(),
                bins=40,
                alpha=0.6,
                color="blue",
                label="random pairs",
                density=True,
            )
            axs[i].hist(
                target.cpu().numpy(),
                bins=40,
                alpha=0.7,
                color="red",
                label="counterfactual pairs",
                density=True,
            )

            # Customize plot appearance
            axs[i].set_yticks([])
            axs[i].set_title(concept_names[i], fontsize=title_fontsize, pad=15)
            axs[i].tick_params(axis="x", labelrotation=45, labelsize=10)
            axs[i].grid(True, alpha=0.3)

            # Remove unnecessary spines
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["left"].set_visible(False)
        else:
            axs[i].axis("off")

    # Add legend to the last subplot
    handles, labels = axs[0].get_legend_handles_labels()
    if n_plots < len(axs):
        legend_ax = axs[n_plots]
    else:
        legend_ax = axs[-1]
    legend_ax.legend(handles, labels, loc="center", fontsize=12, frameon=False)
    legend_ax.axis("off")

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, "concept_direction_LOO.png"),
        bbox_inches="tight",
        dpi=300,
    )
    print(f"Saved figure to {os.path.join(save_dir, 'concept_direction_LOO.png')}")
    plt.show()


def generate_text(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    prompt_type: str = "bare",
    max_new_tokens: int = 100,
    batch_size: int = 16,
    max_save_step: int = 1000,
) -> list[str]:
    prompt_templates = {
        "reflection": "Consider the deeper meaning behind this: {}\nThis text reflects values related to:",
        "analysis": "Analyzing the underlying principles in: {}\nThe core values expressed here are:",
        "implicit": "Looking at the implicit meaning of: {}\nThe fundamental values shown are:",
        "explicit": "What values are represented in this text: {}\nThe text demonstrates values of:",
        "bare": "{}",
        "theme": "What is the main theme in this text: {}\nKey themes include:",
        "topic": "Identify the primary topics discussed in: {}\nMain topics covered are:",
    }

    if prompt_type not in prompt_templates:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Available types: {list(prompt_templates.keys())}"
        )

    template = prompt_templates[prompt_type]
    formatted_prompts = [template.format(prompt) for prompt in prompts]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    generated_texts = []

    # max_save_stepをバッチサイズで考慮
    effective_max_save_step = max_save_step // batch_size * batch_size

    for i in tqdm(
        range(0, len(formatted_prompts), batch_size),
        desc="Generating texts",
        total=len(formatted_prompts) // batch_size
        + bool(len(formatted_prompts) % batch_size),
    ):
        if i > effective_max_save_step:
            break
        batch_prompts = formatted_prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(
            inputs.input_ids, skip_special_tokens=True
        )

        batch_generated = [
            output[len(input_text) :].strip()
            for output, input_text in zip(decoded_outputs, decoded_inputs, strict=False)
        ]
        generated_texts.extend(batch_generated)

    return generated_texts


def save_generation_results(
    pairs: list[list[str]],
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_path: str,
    concept_name: str,
    prompt_type: str,
    max_new_tokens: int = 100,
    batch_size: int = 8,
    max_save_step: int = 1000,
) -> None:
    concept_texts = [pair[0] for pair in pairs]
    non_concept_texts = [pair[1] for pair in pairs]

    generated_concept_texts = generate_text(
        model,
        tokenizer,
        concept_texts,
        prompt_type,
        max_new_tokens,
        batch_size,
        max_save_step,
    )
    generated_non_concept_texts = generate_text(
        model,
        tokenizer,
        non_concept_texts,
        prompt_type,
        max_new_tokens,
        batch_size,
        max_save_step,
    )

    results = [
        {
            "concept_text": c_text,
            "non_concept_text": nc_text,
            "generated_text_concept": gc_text,
            "generated_text_non_concept": gnc_text,
        }
        for c_text, nc_text, gc_text, gnc_text in zip(
            concept_texts,
            non_concept_texts,
            generated_concept_texts,
            generated_non_concept_texts,
            strict=False,
        )
    ]

    output_path = output_path.split(".json")[0] + f"_{concept_name}.json"
    print(f"{output_path=}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    parser.add_argument("--dataset_type", type=str, default="valuenet")
    parser.add_argument("--concept_direction", type=str, default="pos2neg")
    parser.add_argument("--norm_type", type=str, default="base")
    parser.add_argument("--prompt_type", type=str, default="reflection")
    parser.add_argument("--embedding_strategy", type=str, default="last")
    parser.add_argument("--target_layer", type=int, default=16)
    parser.add_argument("--num_sample", type=int, default=1000)
    parser.add_argument("--device_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    dataset_type = args.dataset_type
    concept_direction = args.concept_direction
    norm_type = args.norm_type
    prompt_type = args.prompt_type
    embedding_strategy = args.embedding_strategy
    target_layer = int(args.target_layer)
    device_id = args.device_id

    # Basic config
    model_name = model_path.split("/")[1].lower()
    num_sample = 1000
    counterfactual_output_path = f"/home/itai/research/PersonalValuesGeometry/data/ValueNet/schwartz/{concept_direction}/{norm_type}"
    analyzed_figure_path = f"/home/itai/research/PersonalValuesGeometry/figures/embeddings/{model_name}/layer{target_layer}/{dataset_type}/{concept_direction}/{norm_type}/{prompt_type}"
    generation_random_output_path = f"/home/itai/research/PersonalValuesGeometry/generated/embeddings/{model_name}/layer{target_layer}/{dataset_type}/{concept_direction}/{norm_type}/{prompt_type}/random.json"
    generation_counterfactual_output_path = f"/home/itai/research/PersonalValuesGeometry/generated/embeddings/{model_name}/layer{target_layer}/{dataset_type}/{concept_direction}/{norm_type}/{prompt_type}/counterfactual.json"
    random_txt_path = f"/home/itai/research/PersonalValuesGeometry/data/ValueNet/schwartz/random_pairs/{norm_type}/random_1000_pairs.txt"

    inner_product_matrix_path = f"/home/itai/research/PersonalValuesGeometry/matrices/{model_name}/layer{target_layer}/{dataset_type}/{concept_direction}/{norm_type}/{prompt_type}"

    print("*===== Args =====*")
    print(f"{model_path=}")
    print(f"{dataset_type=}")
    print(f"{concept_direction=}")
    print(f"{norm_type=}")
    print(f"{prompt_type=}")
    print(f"{embedding_strategy=}")
    print(f"{target_layer=}")
    print(f"{num_sample=}")
    print("*===============*")

    # Load model and tokenizer
    device = torch.device(f"cuda:{device_id}")
    if "8B" in model_path:
        print("!! Load model in fp8")
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device}),
        load_in_8bit=True,  # fp8での読み込みを有効化
        quantization_config=bnb.options.ServerConfig(
            compress_statistics=True,
            only_use_fp8=True  # fp8のみを使用
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": device})
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Basic model config
    num_hidden_layers = model.config.num_hidden_layers
    unembedding = model.lm_head.weight.detach()

    # Pairデータの取得
    with open("experiments/counterfactual_pair.txt") as f:
        filenames = [line.strip() for line in f.readlines()]
    # #TODO: get_sequence_pairsとget_counterfactual_pairsはファイル読み込み機構が重複しているので統合する
    random_pairs = get_sequence_pairs(random_txt_path, int(num_sample))
    random_base_sequences, random_target_sequences = get_counterfactual_pairs(
        random_txt_path, prompt_type=prompt_type, num_sample=int(num_sample)
    )
    print("Compute random base embbedings ...")
    random_base_embeddings = get_hidden_layer_n(
        model,
        tokenizer,
        random_base_sequences,
        n_layer=num_hidden_layers,
        batch_size=4,
        embedding_strategy="last",
    )

    print("Compute random target embbedings ...")
    random_target_embeddings = get_hidden_layer_n(
        model,
        tokenizer,
        random_target_sequences,
        n_layer=num_hidden_layers,
        batch_size=4,
        embedding_strategy="last",
    )
    random_vector, random_diff_embeddings = get_concept_vector(
        random_base_embeddings, random_target_embeddings
    )
    random_inner_product_LOO = compute_inner_product_LOO(random_diff_embeddings)

    # Counterfactual pairの生成
    all_inner_product_LOO = []
    all_concept_vectors = []
    concept_names = []
    for _, filename in enumerate(filenames):
        concept_name = get_concpet_name_from_filename(filename)
        concept_names.append(concept_name)
        print(f"Compute {concept_name} ...")
        base_sequences, target_sequences = get_counterfactual_pairs(
            filename, prompt_type=prompt_type, num_sample=int(num_sample)
        )
        base_embeddings = get_hidden_layer_n(
            model=model,
            tokenizer=tokenizer,
            sequences=base_sequences,
            n_layer=target_layer,
            batch_size=4,
            embedding_strategy=embedding_strategy,
        )
        target_embeddings = get_hidden_layer_n(
            model=model,
            tokenizer=tokenizer,
            sequences=target_sequences,
            n_layer=target_layer,
            batch_size=4,
            embedding_strategy=embedding_strategy,
        )

        concept_vector, diff_embeddings = get_concept_vector(
            base_embeddings, target_embeddings
        )
        all_concept_vectors.append(concept_vector)
        concept_inner_product_LOO = compute_inner_product_LOO(diff_embeddings)
        all_inner_product_LOO.append(concept_inner_product_LOO)

        # generate text
        # save_generation_results(
        #     pairs=random_pairs,
        #     model=model,
        #     tokenizer=tokenizer,
        #     output_path=generation_random_output_path,
        #     concept_name=concept_name,
        #     prompt_type=prompt_type,
        #     max_new_tokens=100,
        #     batch_size=8,
        #     max_save_step=4,
        # )

    # Save matrix
    inner_product_matrix_path = os.path.join(
        inner_product_matrix_path, "concept_vector.npy"
    )
    os.makedirs(os.path.dirname(inner_product_matrix_path), exist_ok=True)
    np.save(
        inner_product_matrix_path,
        [concept_vector.cpu().numpy() for concept_vector in all_concept_vectors],
    )

    # Visualize LOO histograms
    show_histogram_LOO(
        all_inner_product_LOO=all_inner_product_LOO,
        random_inner_product_LOO=random_inner_product_LOO,
        concept_names=concept_names,
        save_dir=analyzed_figure_path,
        cols=4,
        title_fontsize=12,
    )
