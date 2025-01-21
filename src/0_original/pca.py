import os
import re

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def get_concpet_name_from_filename(filename: str):
    pattern = r"\[(.*?)\("
    match1 = re.search(pattern, filename.split("/")[-1])
    concept_name = match1.group(1)
    return concept_name


all_inner_product_LOO_path = "/home/itai/research/PersonalValuesGeometry/matrices/llama-3.2-3b-instruct/layer17/valuenet/pos2neg/base/explicit/all_inner_product_LOO.npy"
all_inner_product_LOO = np.load(all_inner_product_LOO_path)

with open(
    "/home/itai/research/PersonalValuesGeometry/experiments/counterfactual_pair.txt"
) as f:
    filenames = [line.strip() for line in f.readlines()]
concept_names = [get_concpet_name_from_filename(filename) for filename in filenames]


num_layers = [i for i in range(22, 29)]
sentence_structures = ["base", "norm_sentence_structure"]
prompt_types = ["topic"]

for sentence_structure in sentence_structures:
    print(f"=== {sentence_structure} ===")
    for prompt_type in prompt_types:
        print(f"=== {prompt_type} ===")
        for i_layer in num_layers:
            print(f"=== Layer {i_layer} ===")
            all_concept_vectors_path = f"/home/itai/research/PersonalValuesGeometry/matrices/llama-3.2-3b-instruct/layer{i_layer}/valuenet/pos2neg/{sentence_structure}/{prompt_type}/concept_vector.npy"
            try:
                all_concept_vectors = np.load(all_concept_vectors_path)
            except:
                print(f"Could not find {all_concept_vectors_path}")
                continue

            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(all_concept_vectors)

            concept_data = {
                "Name": concept_names,
                "PC1": vectors_2d[:, 0],
                "PC2": vectors_2d[:, 1],
            }

            fig = go.Figure()

            for name, x, y in zip(
                concept_data["Name"],
                concept_data["PC1"],
                concept_data["PC2"],
                strict=False,
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[0, x],
                        y=[0, y],
                        mode="lines+markers+text",
                        text=[None, name],
                        textposition="top center",
                        marker=dict(size=8),
                        line=dict(color="rgba(50, 150, 250, 0.7)", width=2),
                    )
                )

            fig.update_layout(
                title=f"Concept Vectors\nLayer: {i_layer}, Prompt: {prompt_type}",
                xaxis=dict(
                    title="",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="rgba(0, 0, 0, 0.3)",
                ),
                yaxis=dict(
                    title="",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="rgba(0, 0, 0, 0.3)",
                ),
                width=800,
                height=800,
                showlegend=False,
                template="plotly_white",
            )

            fig.show()
            save_path = f"/home/itai/research/PersonalValuesGeometry/figures/geometry/{sentence_structure}/{prompt_type}/layer_{i_layer}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path, format="png", scale=2)
