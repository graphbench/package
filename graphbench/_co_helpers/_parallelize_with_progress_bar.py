from collections.abc import Callable, Iterable

import torch
from tqdm import tqdm


def parallelize_with_progress_bar(function: Callable, inputs: Iterable, num_workers: int = 0) -> list:
    """
    Executes a function on the given inputs in parallel while displaying a progress bar.

    If `num_workers <= 0`, then the number of workers is set to the number of CPUs.
    """
    outputs = []
    num_workers = num_workers if num_workers > 0 else torch.multiprocessing.cpu_count()

    with torch.multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap(function, inputs), total=len(inputs)):
            outputs.append(result)

    return outputs
