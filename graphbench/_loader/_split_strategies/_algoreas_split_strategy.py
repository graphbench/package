from graphbench._helpers import split_dataset
from ._split_strategy import DatasetFactory, SplitStrategy, TrainValTestSet


class AlgoReasSplitStrategy(SplitStrategy):
    def build(self, factory: DatasetFactory, dataset_name: str) -> TrainValTestSet:
        if "sizegen" in dataset_name:
            return {
                "train": None,
                "val": None,
                "test": factory(dataset_name, "test", dataset_name),
            }

        dataset = factory(dataset_name, "train", f"{dataset_name}_16")
        train_dataset, valid_dataset, _ = split_dataset(dataset, 0.99, 0.01, 0)
        test_suffix = "64" if "flow" in dataset_name else "128"
        test_dataset = factory(dataset_name, "test", f"{dataset_name}_{test_suffix}")
        return {
            "train": train_dataset,
            "val": valid_dataset,
            "test": test_dataset,
        }
