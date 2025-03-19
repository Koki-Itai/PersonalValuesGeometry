import os
from pathlib import Path

from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """各種パスの設定"""

    base_dir: str = Field(default="/home/itai/research/PersonalValuesGeometry")
    values_file: str = Field(default="values.txt")

    def __init__(self, model_config: ModelConfig, data_config: DataConfig, **kwargs):
        super().__init__(**kwargs)
        self.model_config = model_config
        self.data_config = data_config

    @property
    def path_structure(self) -> str:
        return (
            f"{self.model_config.model_name}/embeddings/layer{self.model_config.target_layer}/"
            f"{self.data_config.dataset_type}/{self.data_config.concept_direction_type}/"
            f"{self.data_config.norm_type}/{self.data_config.prompt_type}"
        )

    @property
    def counterfactual_output_path(self) -> str:
        return f"{self.base_dir}/data/ValueNet/schwartz/{self.data_config.concept_direction_type}/{self.data_config.norm_type}"

    @property
    def analyzed_figure_path(self) -> str:
        return f"{self.base_dir}/figures/{self.path_structure}"

    @property
    def generation_random_output_path(self) -> str:
        return f"{self.base_dir}/generated/{self.path_structure}/random.json"

    @property
    def generation_counterfactual_output_path(self) -> str:
        return f"{self.base_dir}/generated/{self.path_structure}/counterfactual.json"

    @property
    def random_txt_path(self) -> str:
        return f"{self.base_dir}/data/ValueNet/schwartz/random_pairs/{self.data_config.norm_type}/random_1000_pairs.txt"

    @property
    def inner_product_matrix_path(self) -> str:
        return f"{self.base_dir}/matrices/{self.path_structure}"

    def get_counterfactual_data_path(self, value_str: str) -> str:
        return (
            f"{self.base_dir}/datasets/ValueNet/schwartz/{self.data_config.concept_direction_type}/"
            f"{self.data_config.norm_type}/[{value_str}(Positive) - {value_str}(Negative)].txt"
        )

    def ensure_dirs_exist(self):
        """必要なディレクトリが存在することを確認"""
        dirs = [
            os.path.dirname(self.analyzed_figure_path),
            os.path.dirname(self.generation_random_output_path),
            os.path.dirname(self.inner_product_matrix_path),
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
