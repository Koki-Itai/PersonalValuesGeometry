import torch
from sklearn.decomposition import PCA


def apply_pca(embeddings, n_components=10):
    """Apply PCA to reduce dimensions of embeddings."""
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return torch.tensor(reduced_embeddings)
