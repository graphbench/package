from torch_geometric.data import Dataset


def split_dataset(dataset: Dataset, train: float, val: float, test: float) -> tuple[Dataset, Dataset, Dataset]:
    """
    Deterministically splits a dataset into training, validation, and test sets based on the specified ratios.

    If `train`, `val`, and `test` do not sum up to 1, they will be normalized.
    """
    # normalize train, val, test to sum up to 1
    total = train + val + test
    train /= total
    val /= total
    test /= total

    num_graphs = len(dataset)
    train_end = int(train * num_graphs)
    val_end = train_end + int(val * num_graphs)

    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]

    return train_dataset, val_dataset, test_dataset
