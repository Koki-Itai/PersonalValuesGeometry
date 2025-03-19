from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_type: str = Field(default="valuenet")
    concept_direction_type: str = Field(default="pos2neg")
    norm_type: str = Field(default="base")
    prompt_type: str = Field(default="reflection")
    num_sample: int = Field(default=1000)
