import numpy as np
import torch


def calcurate_concept_matrics_rank(
    concept_matrics: torch.Tensor, epsilon: float
) -> torch.Tensor:
    concept_matrics = concept_matrics.numpy()
    U, S, Vt = np.linalg.svd(concept_matrics)
    rank = np.sum(S > epsilon)
    return rank
