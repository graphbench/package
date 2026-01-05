"""
evaluator.py
----------------
Utility class to evaluate model outputs for tasks supported by
GraphBench. The `Evaluator` class centralizes selection of metrics and
computes task-specific scores such as classification accuracy, F1,
regression metrics, and specialized scores used by
benchmarks (e.g., ClosedGap, ChipDesignScore, Weather_MSE).

"""

import os

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops, unbatch, unbatch_edge_index

from graphbench.helpers.utils import VectorizedCircuitSimulator
from graphbench.weatherforecasting_helpers.losses import (
    compute_latitude_weights,
    compute_pressure_level_weights,
    get_default_pressure_levels,
    get_variable_weights,
    masked_loss,
)


class Evaluator():
    """Select and compute metrics for specified benchmark tasks.

    Args:
        name (str): The named benchmark. The implementation reads
            `master.csv` in the module directory and expects a row for
            `name` containing `task` and `metric` columns.

    """

    def __init__(self, name):
        self.name = name 
        self.csv_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)

        self.task = self.csv_info.loc[self.name]['task']
        self.metric = self.csv_info.loc[self.name]['metric'].split(';')


    def _check_input(self, y_pred, y_true=None, batch=None):
        if batch is not None:
            if isinstance(y_pred, np.ndarray):
                y_pred = torch.from_numpy(y_pred)
            if not isinstance(y_pred, torch.Tensor) and not isinstance(y_pred, np.ndarray):
                raise ValueError(f"y_pred must be a torch.Tensor or numpy.ndarray. Got {type(y_pred)}.")
            return y_pred, batch
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)

        if y_pred.size(0) != y_true.size(0):
            raise ValueError(f"y_pred and y_true must have the same number of samples. Got {y_pred.size(0)} and {y_true.size(0)}.")
        
        if not isinstance(y_pred, torch.Tensor) and not isinstance(y_pred, np.ndarray):
            raise ValueError(f"y_pred must be a torch.Tensor or numpy.ndarray. Got {type(y_pred)}.")
        if not isinstance(y_true, torch.Tensor) and not isinstance(y_true, np.ndarray):
            raise ValueError(f"y_true must be a torch.Tensor or numpy.ndarray. Got {type(y_true)}.")
        
        if not y_true.ndim == 2 or not y_pred.ndim == 2:
            raise RuntimeError('y_true and y_pred are supposed to be 2-dim arrays, {}-dim array given'.format(y_true.ndim))

        return y_true, y_pred

    def _get_metric_from_name(self, metric_name):
        """Return a callable that computes the named metric.

        The callable returned generally accepts `(y_pred, y_true)` and
        returns a scalar tensor or numeric value. Some specialized
        metrics accept and ignore extra arguments.
        """
        metric_dict = {
            'ACC': self.get_acc(),
            'F1': self.get_f1(),
            'spearman_r_0': self.get_spearman(0),
            'spearman_r_1': self.get_spearman(1),
            'spearman_r_2': self.get_spearman(2),
            'r2_0': self.get_r2(0),
            'r2_1': self.get_r2(1),
            'r2_2': self.get_r2(2),
            'MSE': self.get_mse(),
            'MAE': self.get_mae(),
            'RMSE': self.get_rmse(),
            'RSE': self.get_rse(),
            'ChipDesignScore': self.get_chip_design_score(),
            'Weather_MSE': self.get_weather_mse(),
            'ClosedGap': self.get_closed_gap(),
            'MisSize': self.get_mis_size(),
            'MaxCutSize': self.get_max_cut_size(),
            'NumColorsUsed': self.get_num_colors_used(),
            
        }
        if metric_name in metric_dict:
            return metric_dict[metric_name]
        else:
            raise ValueError(f"Metric {metric_name} not recognized.")


    def _get_metric(self):
        print(f"Using metric: {self.metric} for task: {self.task}")
        # Check length of metric list and return either single callable
        # or list of callables.
        if len(self.metric) == 1:
            return self._get_metric_from_name(self.metric[0])
        else:
            metric_list =[]
            for metric in self.metric:
                metric_list.append(self._get_metric_from_name(metric))
            return metric_list
        

    def evaluate(self, y_pred, y_true=None, batch=None):
        """
        Computes the selected metric(s) for the given predictions and true values.
        Expects tensors of shape (N, K) where N is the number of samples (nodes or graphs) and K is either the number of classes (for multiclass tasks) or the number of tasks to be evaluated. 
        If multiple batches are computed before metric evaluation, they should be concatenated along the first axis. 
        In case of specialized metrics that require batch information (e.g., unsupervised tasks), the `batch` argument should be provided instead of y_true.
        Returns a single scalar value if one metric is selected, or a list of scalar values if multiple metrics are selected.

        :param y_pred: predicted values as a torch tensor or numpy array of shape (N,K)
        :param y_true: true values as a torch tensor or numpy array of shape (N,K) or (N,1), defaults to None 
        :param batch: optional batch information for unsupervised tasks, defaults to None
        """
        metric = self._get_metric()
        if batch is not None:
            y_pred, batch = self._check_input(y_pred, y_true, batch)
            if isinstance(metric, list):
                return [met(y_pred, batch).item() for met in metric]
            return metric(y_pred, batch).item()

        y_true, y_pred = self._check_input(y_pred, y_true)

        if isinstance(metric, list):
            return [met(y_pred, y_true).item() for met in metric]
        return metric(y_pred, y_true).item()
                
    def get_f1(self):
        """Return a callable computing binary F1.

        Returns:
            Callable[[Tensor, Tensor], Tensor]: Metric callable taking
            `(y_pred, y_true)`.
        """
        f1 = torchmetrics.F1Score(task="binary")
        return lambda x, y: f1(x, y)

    def get_acc(self):
        """Return a callable computing binary accuracy."""
        acc = torchmetrics.Accuracy(task="binary")
        return lambda x, y: acc(x, y)

    def get_spearman(self, index):
        """Return a spearman correlation callable for the given output index."""
        spearman = torchmetrics.SpearmanCorrCoef()
        return lambda x, y: spearman(x[:,index], y[:,index])
    
    def get_r2(self, index):
        """Return an R2 score callable for the given output index."""
        r2 = torchmetrics.R2Score()
        return lambda x, y: r2(x[:,index], y[:,index])

    def get_closed_gap(self):
        """Return a callable computing ClosedGap.

        Note: This metric expects `y_true` shaped (N, K) of runtimes or
        costs per algorithm, and `y_pred` shaped (N, K) of scores or
        probabilities used to select the algorithm.
        """
        return lambda y_pred, y_true: self._get_closed_gap(y_pred, y_true)

    def get_chip_design_score(self):
        """Return a callable computing ChipDesignScore."""
        return lambda y_pred, y_true: self._get_chip_design_score(y_pred, y_true)

    def get_weather_mse(self):
        """Return a callable computing Weather_MSE."""
        return lambda y_pred, y_true: self._get_weather_mse(y_pred, y_true)

    def get_mis_size(self):
        """Return a callable computing MisSize.

        Note: This callable expects `(x, batch)` rather than the standard
        `(y_pred, y_true)` signature.
        """
        return lambda x, batch, dec_length=300, num_seeds=1: self._get_mis_size(x, batch, dec_length=dec_length, num_seeds=num_seeds)

    def get_max_cut_size(self):
        """Return a callable computing MaxCutSize.

        Note: This callable expects `(x, batch)` rather than the standard
        `(y_pred, y_true)` signature.
        """
        return lambda x, batch: self._get_max_cut_size(x, batch)

    def get_num_colors_used(self):
        """Return a callable computing NumColorsUsed.

        Note: This callable expects `(x, batch)` rather than the standard
        `(y_pred, y_true)` signature.
        """
        return lambda x, batch, num_seeds=1: self._get_num_colors_used(x, batch, num_seeds=num_seeds)
    
    def _get_closed_gap(self,y_pred, y_true, inference_times=None):

        # Compute weighed predicted best performance
        # predicted_best_performance = torch.sum(y_pred * y_true, dim=1)

        # compute highest predicted probability best performance
        _, predicted_class_indices = torch.max(y_pred, dim=1)
        predicted_best_performance = y_true[
            torch.arange(y_true.size(0)), predicted_class_indices
        ]

        if inference_times is not None:
            inference_times = torch.cat(inference_times, dim=0)
            predicted_best_performance += inference_times

        # Compute virtual best performance
        virtual_best_performance = torch.min(y_true, dim=1).values

        # Compute single best performance (from index 0)
        single_best_performance = y_true[:, 0]

        # Compute the closed gap
        numerator = torch.mean(single_best_performance) - torch.mean(
            predicted_best_performance
        )
        denominator = torch.mean(single_best_performance) - torch.mean(
            virtual_best_performance
        )

        # Avoid division by zero
        if denominator == 0:
            return torch.tensor(float("nan"))

        closed_gap = numerator / denominator

        return closed_gap
    

    def _get_chip_design_score(self,y_pred, y_true):
        """Compute a chip design equivalence score.

        This method expects `y_pred` and `y_true` to be sequences of
        circuit-like data objects. For each pair it attempts to simulate
        truth-tables using the VectorizedCircuitSimulator class and compares
        outputs. The returned score is in >= 0 with 100 as the score obtained for providing the reference solution.
        """
        if len(y_pred) != len(y_true):
            return 0.0
            
        total_score = 0.0
        N = len(y_pred)
        
        for pred_circuit, target_circuit in zip(y_pred, y_true):
            try:
                # Extract input/output counts from target circuit
                if hasattr(target_circuit, 'num_inputs') and hasattr(target_circuit, 'num_outputs'):
                    num_inputs = target_circuit.num_inputs
                    num_outputs = target_circuit.num_outputs
                else:
                    # Extract from node features using proper extraction logic
                    num_inputs, num_outputs = self.extract_input_output_counts(target_circuit.x)
                
                # Set num_inputs and num_outputs on both circuits
                pred_circuit.num_inputs = num_inputs
                pred_circuit.num_outputs = num_outputs
                target_circuit.num_inputs = num_inputs  
                target_circuit.num_outputs = num_outputs
                
                # Simulate both circuits
                pred_sim = VectorizedCircuitSimulator(pred_circuit)
                target_sim = VectorizedCircuitSimulator(target_circuit)
                
                pred_truth = pred_sim.simulate_all_patterns()
                target_truth = target_sim.simulate_all_patterns()
                
                # Check equivalence and get score
                sample_score = self._equivalence_score(
                    pred_truth, target_truth,
                    pred_circuit.x.shape[0], 
                    target_circuit.x.shape[0] 
                )
                
                total_score += sample_score
                
            except Exception as e:
                # Skip problematic samples 
                print(f"Skipping sample due to error: {e}")
                continue
        
        return (100.0 * total_score) / N if N > 0 else 0.0
    
    def _extract_truth_vectors(self,truth_vectors, num_inputs, num_outputs):
        """Fast truth vector extraction with numpy operations.

        Convert arrays with possible -1 padding to a compact boolean
        matrix of shape `(num_outputs, 2**num_inputs)`.
        Returns `None` on invalid input.
        """
        expected_length = 2**num_inputs
        result = np.zeros((num_outputs, expected_length), dtype=np.uint8)
        
        for output_idx, truth_vector in enumerate(truth_vectors):
            # Find length (-1 padding)
            length = 0
            for val in truth_vector:
                if val == -1:
                    break
                length += 1
            
            if length != expected_length:
                return None  # Invalid truth vector
            
            result[output_idx] = truth_vector[:length]
        
        return result


    def _extract_input_output_counts(self,x):
        """Extract the number of input and output nodes from `x`.

        The method assumes `x` has three columns encoding node types
        as a one-hot vector: [AND, INPUT, OUTPUT]. It counts rows
        matching the `INPUT` and `OUTPUT` patterns.
        
        Args:
            x (Tensor): Node feature tensor of shape `(num_nodes, 3)`.

        Returns:
            tuple: (num_inputs, num_outputs)
        """
        if x.shape[1] != 3:
            raise ValueError(f"Expected node features with 3 columns [AND, INPUT, OUTPUT], got {x.shape[1]}")
        
        # Count input nodes: [0, 1, 0]
        input_mask = (x[:, 1] == 1) & (x[:, 0] == 0) & (x[:, 2] == 0)
        num_inputs = input_mask.sum().item()
        
        # Count output nodes: [0, 0, 1]  
        output_mask = (x[:, 2] == 1) & (x[:, 0] == 0) & (x[:, 1] == 0)
        num_outputs = output_mask.sum().item()
        
        return num_inputs, num_outputs


    def _equivalence_score(self, predicted_truth_vectors, original_truth_vectors, num_nodes_generated, num_nodes_test):
        if np.array_equal(predicted_truth_vectors, original_truth_vectors):
            # equivalence
            if num_nodes_generated > 0:
                return num_nodes_test / num_nodes_generated
            else:
                return 0.0
        else:
            # No match
            return 0.0
        
    def _get_mis_size(self, x: Tensor, batch: Batch, dec_length: int = 300, num_seeds: int = 1) -> Tensor:
        batch = self.mis_decoder(x, batch, dec_length, num_seeds)

        data_list = batch.to_data_list()

        size_list = [data.is_size for data in data_list]

        return Tensor(size_list).mean()


    def _mis_decoder(self, x: Tensor, batch: Batch, dec_length: int = 300, num_seeds: int = 1) -> Batch:
        """Decode MIS scores from model outputs using greedy selection.

        This helper applies a greedy construction over node score logits to
        produce an approximate maximum independent set size for each
        graph in the batch.
        """
        x = torch.sigmoid(x)
        data_list = batch.to_data_list()
        x_list = unbatch(x, batch.batch)

        for data, x_data in zip(data_list, x_list):
            is_size_list = []

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

                is_size_list.append(c.sum())

            data.is_size = max(is_size_list)

        return Batch.from_data_list(data_list)


    def _get_max_cut_size(self, x: Tensor, data: Batch) -> Tensor:
        """Compute average Max-Cut size from binary node assignments.

        Expects `x` as a per-node real-valued tensor; values > 0 are
        treated as one partition, <= 0 as the other.
        """
        x = (x > 0).float()
        x = (x - 0.5) * 2

        x_list = unbatch(x, data.batch)
        edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

        cut_list = []
        for x, edge_index in zip(x_list, edge_index_list):
            cut_list.append(torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) / 2)

        return Tensor(cut_list).mean()


    # TODO: double-check implementation
    def _get_num_colors_used(self, x: Tensor, batch: Batch, num_seeds: int = 1) -> Tensor:
        """Estimate the number of colors used from decoded color labels.

        This function expects a `graph_coloring_decoder` to populate
        a `colors` attribute on each data object in the batch.
        """
        batch = self.graph_coloring_decoder(x, batch, num_seeds)

        data_list = batch.to_data_list()

        num_colors_used_list = []
        for data in data_list:
            num_colors_used = data.colors.unique().size(0)
            num_colors_used_list.append(num_colors_used)

        return torch.tensor(num_colors_used_list).mean(dtype=torch.float)
    
    def _get_weather_mse(self, y_pred, y_true):
        grid_variables = [
        '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
        '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature',
        'geopotential', 'u_component_of_wind', 'v_component_of_wind',
        'vertical_velocity', 'specific_humidity'
    ]
        
                # assuming y_true is the data object and not only the prediction tensor
                # TODO: change format of y_true in evaluator call if needed
        return masked_loss(
                        predictions=y_pred,
                        targets=y_true,
                        variable_slices=None,
                        variable_weights=get_variable_weights(grid_variables), 
                        variable_names=grid_variables,
                            latitude_weights=compute_latitude_weights(y_true.grid_lat),
                            pressure_level_weights=compute_pressure_level_weights(get_default_pressure_levels()),
                            )
    
    def get_mse(self):
        """Return a callable computing mean squared error (averaged per-column)."""
        return lambda y_pred, y_true: self._mse(y_pred, y_true)

    def get_rmse(self):
        """Return a callable computing root mean squared error (averaged per-column)."""
        return lambda y_pred, y_true: self._rmse(y_pred, y_true)

    def get_mae(self):
        """Return a callable computing mean absolute error (averaged per-column)."""
        return lambda y_pred, y_true: self._mae(y_pred, y_true)

    def get_rse(self):
        """Return a callable computing relative squared error (averaged per-column)."""
        return lambda y_pred, y_true: self._rse(y_pred, y_true)

    def _mse(self, y_pred, y_true):
        mse_list = []
        for i in range(y_true.shape[1]):

            mse_list.append(((y_true[:,i] - y_pred[:,i])**2).mean())
        return sum(mse_list)/len(mse_list)
    

    def _rmse(self, y_pred, y_true):

        rmse_list = []

        for i in range(y_true.shape[1]):

            rmse_list.append(np.sqrt(((y_true[:,i] - y_pred[:,i])**2).mean()))

        return sum(rmse_list)/len(rmse_list)
    
    def _mae(self, y_pred, y_true):
        mae_list = []
        for i in range(y_true.shape[1]):

            mae_list.append((torch.abs(y_true[:,i] - y_pred[:,i])).mean())
        return sum(mae_list)/len(mae_list)
    
    def _rse(self, y_pred, y_true):
        """Relative squared error (RSE) averaged over columns.

        For each output dimension $i$:
            $$\mathrm{RSE}_i = \frac{\mathbb{E}[(y_i-\hat{y}_i)^2]}{\mathbb{E}[(y_i-\mathbb{E}[y_i])^2]}$$

        Returns NaN if the variance of `y_true[:, i]` is zero.
        """
        rse_vals = []
        for i in range(y_true.shape[1]):
            num = torch.mean((y_true[:, i] - y_pred[:, i]) ** 2)
            denom = torch.var(y_true[:, i], unbiased=False)
            if torch.isclose(denom, torch.tensor(0.0, device=denom.device)):
                rse_vals.append(torch.tensor(float("nan"), device=denom.device))
            else:
                rse_vals.append(num / denom)

        return sum(rse_vals) / len(rse_vals)


    