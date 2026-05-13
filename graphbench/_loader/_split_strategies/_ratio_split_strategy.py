from graphbench._helpers import split_dataset
from ._split_strategy import DatasetFactory, SplitStrategy, TrainValTestSet


class RatioSplitStrategy(SplitStrategy):
    def __init__(self, train: float, valid: float, test: float) -> None:
        self.train_ratio = train
        self.valid_ratio = valid
        self.test_ratio = test

    def build(self, factory: DatasetFactory, dataset_name: str) -> TrainValTestSet:
        dataset = factory(dataset_name, "train", None)
        train_dataset, valid_dataset, test_dataset = split_dataset(
            dataset,
            self.train_ratio,
            self.valid_ratio,
            self.test_ratio,
        )
        return {
            "train": train_dataset,
            "val": valid_dataset,
            "test": test_dataset,
        }
