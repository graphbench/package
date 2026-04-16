import os
import random

import numpy as np
import torch


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

class SpearmanR(object):

    def _rankdata_torch(self,x: torch.Tensor) -> torch.Tensor:
        """
        Compute average ranks (1-based) for 1D tensor x, matching scipy.stats.rankdata(method='average').
        Works on CPU or GPU tensors.
        Returns float tensor of same device and shape as x.
        """
        if x.ndim != 1:
            raise ValueError("rankdata expects a 1D tensor")

        n = x.numel()
        if n == 0:
            return x.to(dtype=torch.float64)

        # sort values and get sorted indices
        sorted_vals, sorted_idx = torch.sort(x)
        # counts of equal consecutive values in the sorted array
        unique_vals, counts = torch.unique_consecutive(sorted_vals, return_counts=True)

        # compute group start/end indices (1-based ranks)
        counts = counts.to(dtype=torch.long)
        ends = torch.cumsum(counts, dim=0)             # end positions (1-based)
        starts = ends - counts + 1                     # start positions (1-based)

        # average rank for each group
        avg_ranks_per_group = (starts.to(torch.float64) + ends.to(torch.float64)) / 2.0

        # expand average ranks to match sorted positions
        avg_ranks_sorted = torch.repeat_interleave(avg_ranks_per_group, counts).to(dtype=torch.float64)

        # create ranks in original order
        ranks = torch.empty_like(avg_ranks_sorted, dtype=torch.float64, device=x.device)
        ranks[sorted_idx] = avg_ranks_sorted

        return ranks  # 1..n (float64)


    def _spearmanr(self,pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute Spearman's rank correlation (rho) between 1D tensors pred and truth.
        Returns a scalar torch.Tensor (float64) on the same device as inputs.
        Handles ties by assigning average ranks.
        If either input is (very very close to being) constant, returns 0.0.
        """
        if pred.ndim != 1 or truth.ndim != 1:
            raise ValueError("Inputs must be 1D tensors")
        if pred.numel() != truth.numel():
            raise ValueError("Inputs must have the same length")

        device = pred.device
        # compute average ranks (1-based) in double precision
        r_pred = self._rankdata_torch(pred.to(device)).to(device=device)
        r_true = self._rankdata_torch(truth.to(device)).to(device=device)

        # center
        r_pred = r_pred.to(torch.float64)
        r_true = r_true.to(torch.float64)
        rp_centered = r_pred - r_pred.mean()
        rt_centered = r_true - r_true.mean()

        num = (rp_centered * rt_centered).sum()
        denom = torch.sqrt((rp_centered**2).sum() * (rt_centered**2).sum())

        # guard against divide-by-zero (constant input)
        if denom.abs() < eps:
            print('Found close-to-zero denominator in calculation of spearman correlation coeff. Returning 0.')
            return torch.tensor(0.0, dtype=torch.float64, device=device)

        rho = num / denom
        return rho.to(dtype=torch.float64)


    def __call__(self, pred: torch.Tensor, truth: torch.Tensor):
        return self._spearmanr(pred, truth)
    

class VectorizedCircuitSimulator:
    """Vectorized circuit simulator for equivalence checking"""
    
    def __init__(self, data):
        """Pre-compute circuit structure"""
        self.num_inputs = data.num_inputs
        self.num_outputs = data.num_outputs
        self.num_nodes = data.x.shape[0]
        
        # Pre-compute adjacency structure
        self._build_adjacency_arrays(data)
        
        # Pre-generate all input patterns as numpy array
        self.input_patterns = self._generate_all_input_patterns()
        self.num_patterns = len(self.input_patterns)
    
    def _build_adjacency_arrays(self, data):
        # Initialize adjacency lists
        self.node_inputs = [[] for _ in range(self.num_nodes)]
        self.node_inversions = [[] for _ in range(self.num_nodes)]
        
        # Build from edge data
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy().flatten()
        
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            invert = edge_attr[i] == 1.0
            
            self.node_inputs[dst].append(src)
            self.node_inversions[dst].append(invert)
        
        self.max_inputs = max(len(inputs) for inputs in self.node_inputs) if self.node_inputs else 0
        
        self.input_matrix = np.full((self.num_nodes, self.max_inputs), -1, dtype=np.int32)
        self.inversion_matrix = np.zeros((self.num_nodes, self.max_inputs), dtype=bool)
        self.input_counts = np.zeros(self.num_nodes, dtype=np.int32)
        
        for node in range(self.num_nodes):
            num_inputs = len(self.node_inputs[node])
            self.input_counts[node] = num_inputs
            if num_inputs > 0:
                self.input_matrix[node, :num_inputs] = self.node_inputs[node]
                self.inversion_matrix[node, :num_inputs] = self.node_inversions[node]
    
    def _generate_all_input_patterns(self):
        """Pre-generate all 2^n input patterns"""
        patterns = np.zeros((2**self.num_inputs, self.num_inputs), dtype=np.float32)
        for i in range(2**self.num_inputs):
            # Big-endian bit order
            for j in range(self.num_inputs):
                patterns[i, j] = (i >> (self.num_inputs - 1 - j)) & 1
        return patterns
    
    def simulate_all_patterns(self):
        """Simulation of all input patterns simultaneously"""
        # Initialize values for all patterns and nodes
        values = np.zeros((self.num_patterns, self.num_nodes), dtype=np.float32)
        
        # Set input values for all patterns
        values[:, :self.num_inputs] = self.input_patterns
        
        # Process internal AND gates
        for node_idx in range(self.num_inputs, self.num_nodes):
            num_inputs = self.input_counts[node_idx]
            if num_inputs == 0:
                values[:, node_idx] = 0.0  # Default value for unconnected nodes
                continue
            
            # Get input values for this node across all patterns
            input_indices = self.input_matrix[node_idx, :num_inputs]

            valid_inputs = input_indices >= 0
            if not np.any(valid_inputs):
                values[:, node_idx] = 0.0
                continue
            
            # Get input values
            input_vals = values[:, input_indices[valid_inputs]]
            
            # Apply inversions for valid inputs only
            inversions = self.inversion_matrix[node_idx, :num_inputs][valid_inputs]
            input_vals = np.where(inversions, 1.0 - input_vals, input_vals)
            
            # Compute AND across all inputs for each pattern
            if input_vals.shape[1] == 1:
                values[:, node_idx] = input_vals[:, 0]
            else:
                # AND gate. Multiply all inputs
                values[:, node_idx] = np.prod(input_vals, axis=1)
        
        # Extract output values
        output_start = self.num_nodes - self.num_outputs
        output_values = values[:, output_start:].astype(np.uint8)
        
        return output_values.T
