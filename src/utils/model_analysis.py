import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_hidden_layer_n(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sequences: list[str],
    n_layer: int,
    batch_size: int = 4,
    embedding_strategy: str = "last",
) -> torch.Tensor:
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
                layer_hidden_state = layer_embeddings
                last_token_indices = attention_mask.sum(dim=1) - 1
                batch_embeddings = layer_hidden_state[
                    torch.arange(layer_hidden_state.size(0)), last_token_indices
                ]
                all_embeddings.append(batch_embeddings.cpu())
                del outputs, hidden_states, layer_embeddings, layer_hidden_state
        else:
            raise ValueError(
                f"Invalid embedding_strategy: {embedding_strategy}. Available types: [mean, last]"
            )
    return torch.cat(all_embeddings, dim=0)


def get_concept_vector(diff_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        concept_vector: shape=(hidden_size,)
    """
    diff_embeddings = diff_embeddings / torch.norm(diff_embeddings, dim=1, keepdim=True)
    concept_direction_vector = torch.mean(diff_embeddings, dim=0)
    return concept_direction_vector


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
