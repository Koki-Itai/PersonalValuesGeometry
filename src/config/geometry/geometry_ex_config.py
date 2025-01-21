from abc import ABC

from pydantic import BaseModel, field_validator


class GeometryExperimentsConfig(BaseModel, ABC):
    model_path: str
    dataset_type: str
    concept_direction_type: str
    norm_type: str
    prompt_type: str
    embedding_strategy_type: str
    target_layer: int
    num_sample_pairs_data: int
    device: int | str  # device_id or "auto"
    log_level: str
    log_file: str

    @field_validator("model_path", mode="after")
    def validate_model_path(cls, model_path: str) -> str:
        if not model_path:
            raise ValueError("model_path is required")
        return model_path

    @field_validator("dataset_type", mode="after")
    def validate_dataset_type(cls, dataset_type: str) -> str:
        if not dataset_type:
            raise ValueError("dataset_type is required")
        return dataset_type

    @field_validator("concept_direction_type", mode="after")
    def validate_concept_direction(cls, concept_direction_type: str) -> str:
        if not concept_direction_type:
            raise ValueError("concept_direction_type is required")
        return concept_direction_type

    @field_validator("norm_type", mode="after")
    def validate_norm_type(cls, norm_type: str) -> str:
        if not norm_type:
            raise ValueError("norm_type is required")
        return norm_type

    @field_validator("prompt_type", mode="after")
    def validate_prompt_type(cls, prompt_type: str) -> str:
        if not prompt_type:
            raise ValueError("prompt_type is required")
        return prompt_type

    @field_validator
    def validate_embedding_strategy(cls, embedding_strategy_type: str) -> str:
        if not embedding_strategy_type:
            raise ValueError("embedding_strategy is required")
        return embedding_strategy_type

    @field_validator
    def validate_target_layer(cls, target_layer: int) -> int:
        if target_layer < 0:
            raise ValueError("target_layer must be a non-negative integer")
        return target_layer

    @field_validator
    def validate_num_sample(cls, num_sample_pairs_data: int) -> int:
        if num_sample_pairs_data < 0:
            raise ValueError("num_sample_pairs_data must be a non-negative integer")
        return num_sample_pairs_data

    @field_validator
    def validate_device(cls, device: int | str) -> int | str:
        if not isinstance(device, int) and device != "auto":
            raise ValueError("device must be an integer or 'auto'")
        return device

    @field_validator
    def validate_log_level(cls, log_level: str) -> str:
        if not log_level:
            raise ValueError("log_level is required")
        return log_level

    @field_validator
    def validate_log_file(cls, log_file: str) -> str:
        if not log_file:
            raise ValueError("log_file is required")
        return log_file
