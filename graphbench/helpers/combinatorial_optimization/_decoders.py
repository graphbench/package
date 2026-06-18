import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops, unbatch


# Source: https://github.com/WenkelF/copt/blob/main/utils/metrics.py
def mis_decoder(x: Tensor, batch: Batch, dec_length: int = 300, num_seeds: int = 1) -> list[Tensor]:
    x = torch.sigmoid(x)
    data_list = batch.to_data_list()
    x_list = unbatch(x, batch.batch)
    decoded_solutions: list[Tensor] = []

    for data, x_data in zip(data_list, x_list):
        best_solution = None
        best_size = None

        for seed in range(num_seeds):

            order = torch.argsort(x_data, dim=0, descending=True)
            c = torch.zeros_like(x_data)

            edge_index = remove_self_loops(data.edge_index)[0]
            src, dst = edge_index[0], edge_index[1]

            c[order[seed]] = 1
            for idx in range(seed, min(dec_length, data.num_nodes)):
                c[order[idx]] = 1

                cTWc = torch.sum(c[src] * c[dst])
                if cTWc != 0:
                    c[order[idx]] = 0

            current_size = c.sum()
            if best_size is None or current_size > best_size:
                best_size = current_size
                best_solution = c.clone()

        decoded_solutions.append(best_solution)

    return decoded_solutions


def graph_coloring_decoder(x: Tensor, batch: Batch, num_seeds: int = 1) -> list[Tensor]:
    max_num_colors = x.size(1)
    x = torch.sigmoid(x)
    data_list = batch.to_data_list()
    x_list = unbatch(x, batch.batch)
    decoded_colorings: list[Tensor] = []

    for data, x_data in zip(data_list, x_list):
        edge_index = remove_self_loops(data.edge_index)[0]
        src, dst = edge_index[0], edge_index[1]

        best_colors = None
        min_colors_used = max_num_colors + 1  # upper bound

        for seed in range(num_seeds):
            order = torch.argsort(x_data.max(dim=1).values, descending=True)
            colors = torch.full((data.num_nodes,), -1, dtype=torch.long, device=x_data.device)

            for idx in range(data.num_nodes):
                node = order[(seed + idx) % data.num_nodes]
                # Find available colors for this node
                used = torch.zeros(max_num_colors, dtype=torch.bool, device=x_data.device)
                neighbors = dst[src == node]
                for neighbor in neighbors:
                    c = colors[neighbor]
                    if c >= 0:
                        used[c] = True

                # Assign the available color with the highest score in x_data for this node
                available_color_indices = (~used).nonzero(as_tuple=True)[0]
                max_idx = torch.argmax(x_data[node, available_color_indices])
                colors[node] = available_color_indices[max_idx]

            num_colors_used = colors.unique().size(0)
            if num_colors_used < min_colors_used:
                min_colors_used = num_colors_used
                best_colors = colors

        decoded_colorings.append(best_colors)

    return decoded_colorings
