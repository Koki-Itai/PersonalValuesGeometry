import torch
from sklearn.decomposition import FastICA


def apply_ica(embeddings, n_components=10) -> torch.Tensor:
    """Apply ICA to reduce dimensions of embeddings."""
    ica = FastICA(n_components=n_components)
    reduced_embeddings = ica.fit_transform(embeddings)
    return torch.tensor(reduced_embeddings)
