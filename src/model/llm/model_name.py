from enum import Enum


class ModelName(str, Enum):
    LLAMA_3_1_8B_BASE = "meta-llama/Llama-3.1-8B"
    LLAMA_3_1_8B_INST = "meta-llama/Llama-3.1-8B-instruct"
    LLAMA_3_2_3B_BASE = "meta-llama/Llama-3.2-3B"
    LLAMA_3_2_3B_INST = "meta-llama/Llama-3.2-3B-Instruct"
    GEMMA_2_2B_JPN_IT = "google/gemma-2-2b-jpn-it"

    def is_hf_model(self) -> bool:
        return self in [
            ModelName.LLAMA_3_1_8B_BASE,
            ModelName.LLAMA_3_1_8B_INST,
            ModelName.LLAMA_3_2_3B_BASE,
            ModelName.LLAMA_3_2_3B_INST,
            ModelName.GEMMA_2_2B_JPN_IT,

        ]

    def is_llama_model(self) -> bool:
        return self in [
            ModelName.LLAMA_3_1_8B_BASE,
            ModelName.LLAMA_3_1_8B_INST,
            ModelName.LLAMA_3_2_3B_BASE,
            ModelName.LLAMA_3_2_3B_INST,
        ]
