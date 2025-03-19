import os

import matplotlib.pyplot as plt
import torch


def show_histogram_LOO(
    all_inner_product_LOO: list[torch.Tensor],
    random_inner_product_LOO: torch.Tensor,
    concept_names: list[str],
    save_dir: str,
    cols: int = 4,
    title_fontsize: int = 12,
    is_pca: bool = False,
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
    if is_pca:
        save_dir = os.path.join(save_dir, "pca")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, "concept_direction_LOO.png"),
        bbox_inches="tight",
        dpi=300,
    )
    print(f"Saved figure to {os.path.join(save_dir, 'concept_direction_LOO.png')}")
    plt.show()
