import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
from torch import device as torch_device


def get_concpet_name_from_filename(filename: str):
    pattern = r"\[(.*?)\("
    match1 = re.search(pattern, filename.split("/")[-1])
    concept_name = match1.group(1)
    return concept_name


def run_intervention(
    model: torch.nn.Module,
    tokenizer: torch.nn.Module | object,
    texts: list[str],
    concept_vectors: Tensor,
    concept2idx: dict[str, int],
    intervention_value_name: str = "Achievement",
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    step_alpha: float = 0.1,
    n_generate_tokens: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: torch_device = "cuda",
    verbose: bool = False,
    save_output: bool = True,
) -> np.ndarray:
    intervention_value_idx = concept2idx[intervention_value_name]
    concept_vec = concept_vectors[intervention_value_idx]
    alphas = np.arange(min_alpha, max_alpha + step_alpha, step_alpha)

    if save_output:
        os.makedirs("intervention", exist_ok=True)
        output_path = os.path.join("intervention", f"{intervention_value_name}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {intervention_value_name}\n\n")

    for alpha in alphas:
        results = []
        for text in texts:
            prompt = f"Please write a natural continuation following this text:\n{text}"
            encoded = tokenizer(
                prompt,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                max_length=128,
            ).to(device)

            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            generated_ids = input_ids

            for _ in range(n_generate_tokens):
                with torch.no_grad():
                    outputs = model(
                        generated_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

                    last_hidden_state = outputs.hidden_states[-1]
                    last_token_state = last_hidden_state[:, -1, :]
                    intervened_state = last_token_state + alpha * concept_vec
                    logits = model.lm_head(intervened_state)

                    scaled_logits = logits[0] / temperature
                    sorted_logits, sorted_indices = torch.sort(
                        scaled_logits, descending=True
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    sorted_logits[sorted_indices_to_remove] = float("-inf")

                    probs_to_sample = torch.zeros_like(scaled_logits)
                    probs_to_sample[sorted_indices] = sorted_logits
                    final_probs = F.softmax(probs_to_sample, dim=-1)

                    next_token_id = torch.multinomial(
                        final_probs, num_samples=1
                    ).unsqueeze(0)

                    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones((attention_mask.shape[0], 1), device=device),
                        ],
                        dim=-1,
                    )

            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            original_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            continuation = generated_text[len(original_prompt) :].strip()
            full_text = text + " " + continuation

            results.append({"original": text, "interventioned": full_text})

            if verbose:
                print(f"Alpha: {alpha:.1f}, Text: {text}")
                print(f"Generated: {full_text}")

        if save_output:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"\n## α = {alpha:.1f}\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"### Text {i}\n")
                    f.write(f"**Original**: {result['original']}\n\n")
                    f.write(f"**Interventioned**: {result['interventioned']}\n\n")
                f.write("---\n")
    print("Complete All Generation.")


with open(
    "/home/itai/research/PersonalValuesGeometry/experiments/counterfactual_pair.txt"
) as f:
    filenames = [line.strip() for line in f.readlines()]
concept_names = [get_concpet_name_from_filename(filename) for filename in filenames]

device_id = 2
model_path = "meta-llama/Llama-3.2-3B-Instruct"
device = torch.device(f"cuda:{device_id}")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": device},
)
model = model.to(device)

# 価値観の種類を取得
filenames = []
path = "/home/itai/research/PersonalValuesGeometry/experiments/counterfactual_pair.txt"
with open(path) as f:
    for line in f.readlines():
        filenames.append(line.strip())

concept2idx = {}
idx2concept = {}
pattern = r"\[(.*?)\("
for idx, filename in enumerate(filenames):
    match1 = re.search(pattern, filenames[idx].split("/")[-1])
    concept_name = match1.group(1)
    concept2idx[concept_name] = idx
    idx2concept[idx] = concept_name

i_layer = 26
sentence_structure = "norm_sentence_structure"
prompt_type = "explicit"
all_concept_vectors_path = f"/home/itai/research/PersonalValuesGeometry/matrices/llama-3.2-3b-instruct/layer{i_layer}/valuenet/pos2neg/{sentence_structure}/{prompt_type}/concept_vector.npy"
all_concept_vectors = np.load(all_concept_vectors_path)
all_concept_vectors = torch.tensor(all_concept_vectors).to(device)


device = torch.device(f"cuda:{device_id}")

neutral_texts = [
    "The person completed the task.",
    "They made a decision about the matter.",
    "The group discussed the situation.",
    "telling my friend that it will take 7 years for her to graduate with her double major",
    "My colleague is quitting her job.",
]

for value_name in concept2idx.keys():
    print(f"Value Name: {value_name}")
    run_intervention(
        model=model,
        tokenizer=tokenizer,
        texts=neutral_texts,
        concept_vectors=all_concept_vectors,
        concept2idx=concept2idx,
        intervention_value_name=value_name,
        min_alpha=0.0,
        max_alpha=1.0,
        step_alpha=0.1,
        n_generate_tokens=64,
        temperature=0.9,
        top_p=0.95,
        device=device,
        verbose=False,
        save_output=True,
    )
