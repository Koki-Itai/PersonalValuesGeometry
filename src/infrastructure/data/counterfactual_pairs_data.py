
from model.data import CounterfactualPairsDataModel


class CounterfactualPairsDataService:
    def __init__(self) -> None:
        pass

    def load_sequence_pairs(
        self,
    ) -> list[list[str]]:
        pass

    def load_counterfactual_pairs(
        self,
    ) -> tuple[list[str], list[str]]:
        pass
