import numpy as np
import torch
from pydantic import BaseModel, field_validator
from sklearn.decomposition import PCA


class PCAInput(BaseModel):
    matrix: torch.Tensor | np.ndarray
    n_components: int = 2

    @field_validator("matrix")
    def validate_matrix(cls, v):
        if not isinstance(v, torch.Tensor | np.ndarray):
            raise ValueError("Input must be a torch.Tensor or np.ndarray")
        if len(v.shape) != 2:
            raise ValueError("Input must be a 2D tensor or array")
        if v.shape[1] < 2:
            raise ValueError("Input must have at least 2 features")
        return v

    @field_validator("n_components")
    def validate_n_components(cls, v, values):
        if "matrix" in values:
            if v < 1:
                raise ValueError("n_components must be at least 1")
            if v > values["matrix"].shape[1]:
                raise ValueError(
                    "n_components cannot be larger than the number of features"
                )
        return v

    class Config:
        arbitrary_types_allowed = True  # Allow torch.Tensor


def apply_pca(matrix: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    pca_input = PCAInput(matrix=matrix, n_components=n_components)
    matrix_np = pca_input.matrix.detach().cpu().numpy()

    pca = PCA(n_components=pca_input.n_components)
    reduced_vectors = pca.fit_transform(matrix_np)

    return torch.from_numpy(reduced_vectors).to(matrix.device)


# Usage example
# x = torch.randn(100, 10)  # 100 samples x 10 features
# reduced_x = apply_pca(x, n_components=3)
# print(f"Reduced shape: {reduced_x.shape}")  # [100, 3]
