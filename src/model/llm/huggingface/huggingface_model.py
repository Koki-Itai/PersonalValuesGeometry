import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from model.llm.llm_model import LLMModel
from model.llm.messages import Messages
from model.llm.model_name import ModelName


class HuggingFaceModel(LLMModel):
    @classmethod
    def _load_default(
        cls, model_name: ModelName, device: str | int = "auto"
    ) -> "HuggingFaceModel":
        tokenizer = AutoTokenizer.from_pretrained(model_name.value)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = device
        if isinstance(device, int):
            device_map = {
                "": torch.device("cuda", device) if torch.cuda.is_available() else "cpu"
            }

        model = AutoModelForCausalLM.from_pretrained(
            model_name.value,
            torch_dtype="auto",
            device_map=device_map,
        )
        print(f"model: {model}")
        return cls(model=model, tokenizer=tokenizer)

    @classmethod
    def _load_llama(
        cls, model_name: ModelName, device: str | int = "auto"
    ) -> "HuggingFaceModel":
        tokenizer = AutoTokenizer.from_pretrained(model_name.value)
        print(f"tokenizer: {tokenizer}")

        device_map = device
        if isinstance(device, int):
            device_map = {
                "": torch.device("cuda", device) if torch.cuda.is_available() else "cpu"
            }

        model = LlamaForCausalLM.from_pretrained(
            model_name.value,
            torch_dtype="auto",
            device_map=device_map,
        )
        print(f"model: {model}")
        return cls(model=model, tokenizer=tokenizer)

    @classmethod
    def load(
        cls, model_name: ModelName, device: str | int = "auto"
    ) -> "HuggingFaceModel":
        if model_name.is_llama_model():
            return cls._load_llama(model_name, device)
        else:
            return cls._load_default(model_name, device)

    def _generate_default(self, messages: Messages) -> str:
        self.model.eval()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
        answer = self.tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
        )
        return str(answer)

    def _generate_llama(self, messages: Messages) -> str:
        self.model.eval()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                eos_token_id=terminators,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
        answer = self.tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
        )
        return str(answer)

    def generate(self, model_name: ModelName, messages: Messages) -> str:
        if model_name.is_llama_model():
            return self._generate_llama(messages)
        else:
            return self._generate_default(messages)
