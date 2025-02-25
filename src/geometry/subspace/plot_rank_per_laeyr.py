import os

import matplotlib.pyplot as plt
import pandas as pd

save_dir = "/home/itai/research/PersonalValuesGeometry/experiments_results/geometry/matrics_rank_per_layer"
os.makedirs(save_dir, exist_ok=True)

epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
for epsilon in epsilons:
    concept_rank_results_path = f"/home/itai/research/PersonalValuesGeometry/experiments/concept_matrics_ranks/eps{epsilon}.csv"
    df = pd.read_csv(concept_rank_results_path)

    plt.figure(figsize=(10, 6))

    for concept, group in df.groupby("Concept Name"):
        plt.plot(group["Layer"], group["Rank"], marker="o", label=concept)

    avg_df = df.groupby("Layer")["Rank"].mean().reset_index()
    plt.plot(
        avg_df["Layer"],
        avg_df["Rank"],
        marker="x",
        color="black",
        linestyle="--",
        linewidth=2,
        label="Average",
    )

    plt.xlabel("Layer")
    plt.ylabel("Rank")
    plt.title(f"Rank of concepts in each layer (epsilon={epsilon})")
    plt.legend(title="Concept")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"eps{epsilon}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"All plots saved to {save_dir} directory")
