from pathlib import Path

from config.geometry import GeometryExperimentsConfig
from infrastructure.llm.llm import LLMService
from model.data import DataModel
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SubspaceAnalysis:
    def __init__(
        self,
        llm_service: LLMService,
        config_model: GeometryExperimentsConfig,
        data_model: DataModel,
    ):
        self.llm_service = llm_service
        self.config = config_model
        self.data = data_model

        # Basic Configurations
        ## Input Data
        self.llm_model_name = self.config.model_path.split("/")[1].lower()
        self.counterfactual_pairs_dataset_path = Path(
            "data/ValueNet/schwartz"
        ).joinpath(self.config.concept_direction_type, self.config.norm_type)
        self.random_pairs_dataset_path = Path(
            "data/ValueNet/schwartz/random_pairs"
        ).joinpath(
            self.config.norm_type,
            f"random_{self.config.num_sample_pairs_data}_pairs.json",
        )

        ## Output Data
        self.freq_fig_dist_path = Path("figures/geometry").joinpath(
            f"num_sumple_{self.config.num_sample_pairs_data}",
            self.llm_model_name,
            f"layer{self.config.target_layer}",
            self.config.dataset_type,
            self.config.concept_direction_type,
            self.config.norm_type,
            self.config.prompt_type,
        )
        self.generated_next_tokens_random_pairs_path = Path(
            "generated/geometry/next_tokens"
        ).joinpath(
            self.llm_model_name,
            f"layer{self.config.target_layer}",
            self.config.dataset_type,
            self.config.concept_direction_type,
            self.config.norm_type,
            self.config.prompt_type,
            "random.json",
        )
        self.generated_next_tokens_counterfactuals_pairs_path = Path(
            "generated/geometry/next_tokens"
        ).joinpath(
            self.llm_model_name,
            f"layer{self.config.target_layer}",
            self.config.dataset_type,
            self.config.concept_direction_type,
            self.config.norm_type,
            self.config.prompt_type,
            "counterfactual.json",
        )
        self.embedding_matrix_dir = Path(
            f"matrices/{self.llm_model_name}/layer{self.config.target_layer}/{self.config.dataset_type}/{self.config.concept_direction_type}/{self.config.norm_type}/{self.config.prompt_type}"
        )

        logger.info("*===== Args =====*")
        logger.info(f"Model: {self.config.model_path}")
        

