import torch
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """モデル関連の設定パラメータ"""

    model_path: str = Field(default="meta-llama/Llama-3.2-3B-Instruct")
    device_id: int = Field(default=0)
    embedding_batch_size: int = Field(default=4)
    target_layer: int = Field(default=16)
    embedding_strategy: str = Field(default="last")

    @property
    def model_name(self) -> str:
        return self.model_path.split("/")[1].lower()

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_id}")
