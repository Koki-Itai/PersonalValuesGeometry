import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DataConfig, ModelConfig, PathConfig
from src.utils.model_analysis import (
    compute_inner_product_LOO,
    get_concept_vector,
    get_hidden_layer_n,
)
from src.utils.preprocess_data import get_counterfactual_pairs
from src.utils.visualization import show_histogram_LOO


class ConceptEmbeddingGeometryProcessor:
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        path_config: PathConfig,
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.path_config = path_config

        # モデルの読み込み
        self.model = AutoModelForCausalLM.from_pretrained(model_config.model_path).to(
            model_config.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # モデル情報
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.unembedding = self.model.lm_head.weight.detach()

        # 値のリスト
        self.values_list_str = [line.strip() for line in open(path_config.values_file)]

    def get_embeddings(self, sequences: list[str]) -> torch.Tensor:
        """シーケンスの埋め込みを取得"""
        return get_hidden_layer_n(
            model=self.model,
            tokenizer=self.tokenizer,
            sequences=sequences,
            n_layer=self.model_config.target_layer,
            embedding_strategy=self.model_config.embedding_strategy,
            batch_size=self.model_config.embedding_batch_size,
        )

    def process_pair(
        self, positive_sequences: list[str], negative_sequences: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ポジティブとネガティブのシーケンスペアを処理"""
        # 埋め込みを計算
        positive_embeddings = self.get_embeddings(positive_sequences)
        negative_embeddings = self.get_embeddings(negative_sequences)

        # 差分を計算
        diff_embeddings = negative_embeddings - positive_embeddings

        # LOOで内積を計算
        inner_product_LOO = compute_inner_product_LOO(diff_embeddings=diff_embeddings)

        # 概念ベクトルを計算
        concept_vector = get_concept_vector(diff_embeddings=diff_embeddings)

        return inner_product_LOO, concept_vector

    def process_random_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ランダムなペアを処理"""
        print("[Random Pair] random文書pairを取得 ...")
        random_positive_sequences, random_negative_sequences = get_counterfactual_pairs(
            self.path_config.random_txt_path,
            prompt_type=self.data_config.prompt_type,
            num_sample=self.data_config.num_sample,
        )

        print("[Random Pair] positive/negativeのembeddingを計算 ...")
        return self.process_pair(random_positive_sequences, random_negative_sequences)

    def process_concept_pairs(self) -> list[torch.Tensor]:
        """全ての概念ペアを処理"""
        all_values_inner_product_LOO = []

        for value_str in self.values_list_str:
            print("=" * 20 + f"Value: {value_str}" + "=" * 20)
            print("[Counterfactual Pair] counterfactual文書pairを取得 ...")

            counter_factual_data_path = self.path_config.get_counterfactual_data_path(
                value_str
            )
            concept_positive_sequences, concept_negative_sequences = (
                get_counterfactual_pairs(
                    counter_factual_data_path,
                    prompt_type=self.data_config.prompt_type,
                    num_sample=self.data_config.num_sample,
                )
            )

            print(f"[Concept {value_str} Pair] positive/negativeのembeddingを計算 ...")
            inner_product_LOO, concept_vector = self.process_pair(
                concept_positive_sequences, concept_negative_sequences
            )

            all_values_inner_product_LOO.append(inner_product_LOO)

        return all_values_inner_product_LOO

    def run_analysis(self):
        """全ての分析を実行"""
        # ディレクトリの存在確認
        self.path_config.ensure_dirs_exist()

        # ランダムペアの処理
        random_inner_product_LOO, random_concept_vector = self.process_random_pair()

        # 概念ペアの処理
        all_values_inner_product_LOO = self.process_concept_pairs()

        # LOOヒストグラムの可視化
        show_histogram_LOO(
            all_inner_product_LOO=all_values_inner_product_LOO,
            random_inner_product_LOO=random_inner_product_LOO,
            concept_names=self.values_list_str,
            save_dir=self.path_config.analyzed_figure_path,
            cols=4,
            title_fontsize=12,
            is_pca=False,
        )
