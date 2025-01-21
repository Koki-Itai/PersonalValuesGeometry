from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from model.llm.messages import Messages
from model.llm.model_name import ModelName


class LLMModel(BaseModel, ABC):
    model: Any
    tokenizer: Any

    @abstractmethod
    def generate(self, model_name: ModelName, messages: Messages) -> str:
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_name: ModelName) -> "LLMModel":
        pass
