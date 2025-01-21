import os

from dotenv import load_dotenv
from huggingface_hub import login

from model.llm.huggingface.huggingface_model import HuggingFaceModel
from model.llm.llm_model import LLMModel
from model.llm.model_name import ModelName

load_dotenv()


class LLMService:
    def __init__(self) -> None:
        login(token=os.environ.get("HUGGINGFACE_TOKEN"))

    def load_model(
        self,
        model_name: ModelName,
        device: str | int = "auto",
    ) -> LLMModel:
        if model_name.is_hf_model():
            return HuggingFaceModel.load(model_name, device)
        else:
            raise ValueError("Invalid model name")
