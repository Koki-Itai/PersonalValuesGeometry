from abc import ABC, abstractmethod

from pydantic import BaseModel, field_validator


class DataModel(BaseModel, ABC):
    sequence_pairs: list[list[str]]
    counterfactual_pairs: tuple[list[str], list[str]]

    @abstractmethod
    def get_sequence_pairs(self, num_samples: int) -> list[list[str]]:
        pass

    @abstractmethod
    def get_counterfactual_pairs(
        filename: str, prompt_template: str, num_sample: int
    ) -> tuple[list[str], list[str]]:
        pass

    @field_validator("sequence_pairs", mode="after")
    def validate_sequence_pairs(
        cls, sequence_pairs: list[list[str]]
    ) -> list[list[str]]:
        if not all(isinstance(sequence, list) for sequence in sequence_pairs):
            raise ValueError("sequence_pairs must be a list of lists")
        if not all(isinstance(sequence, str) for sequence in sequence_pairs):
            raise ValueError("sequence_pairs must be a list of lists of strings")
        return sequence_pairs

    @field_validator("counterfactual_pairs", mode="after")
    def validate_counterfactual_pairs(
        cls, counterfactual_pairs: tuple[list[str], list[str]]
    ) -> tuple[list[str], list[str]]:
        if not isinstance(counterfactual_pairs, tuple):
            raise ValueError("counterfactual_pairs must be a tuple")
        if len(counterfactual_pairs) != 2:
            raise ValueError("counterfactual_pairs must be a tuple of length 2")
        if not all(isinstance(sequence, list) for sequence in counterfactual_pairs):
            raise ValueError("counterfactual_pairs must be a tuple of lists")
        if not all(isinstance(sequence, str) for sequence in counterfactual_pairs):
            raise ValueError("counterfactual_pairs must be a tuple of lists of strings")
        return counterfactual_pairs
