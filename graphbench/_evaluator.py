from typing import Callable, Optional, Union, TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
import torchmetrics

from graphbench.helpers.combinatorial_optimization import (
    graph_coloring_decoder,
    max_cut_decoder,
    max_cut_size,
    mis_decoder,
    mis_size,
    num_colors_used,
)
from graphbench._metadata import get_master_df
from graphbench._helpers import VectorizedCircuitSimulator
from graphbench._weatherforecasting_helpers import (
    compute_latitude_weights,
    compute_pressure_level_weights,
    get_default_pressure_levels,
    get_variable_weights,
    masked_loss,
)


# note that this is not the same having the Union inside the Callable, which would be too permissive
_Metric: TypeAlias = Union[
    Callable[[Tensor, Tensor], Tensor],
    Callable[[Tensor, Batch], Tensor],
    Callable[[list[Data], list[Data]], Tensor],
]


class Evaluator():
    """Select and compute metrics for specified benchmark tasks.

    Utility class to evaluate model outputs for tasks supported by
    GraphBench. The `Evaluator` class centralizes selection of metrics and
    computes task-specific scores such as classification accuracy, F1,
    regression metrics, and specialized scores used by
    benchmarks (e.g., ClosedGap, ChipDesignScore, Weather_MSE).

    Args:
        name: The named benchmark. The implementation reads
              `master.csv` in the module directory and expects a row for
              `name` containing `task` and `metric` columns.
    """

    def __init__(self, name: str):
        self.csv_info = get_master_df()

        self.task = self.csv_info.loc[name]['task']
        self.metric = self.csv_info.loc[name]['metric'].split(';')

    def _check_input(
        self,
        y_pred: Union[Tensor, np.ndarray],
        y_true: Optional[Union[Tensor, np.ndarray]] = None,
        batch: Optional[Batch] = None,
    ) -> tuple[Tensor, Union[Tensor, Batch]]:
        if batch is None and y_true is None:
            raise ValueError("Either y_true or batch must be provided.")
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

        return y_pred, y_true

    def _get_metric_from_name(self, metric_name: str) -> _Metric:
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

    def _get_metric(self) -> Union[_Metric, list[_Metric]]:
        print(f"Using metric: {self.metric} for task: {self.task}")
        # Check length of metric list and return either single callable
        # or list of callables.
        if len(self.metric) == 1:
            return self._get_metric_from_name(self.metric[0])
        else:
            metric_list = []
            for metric in self.metric:
                metric_list.append(self._get_metric_from_name(metric))
            return metric_list

    def evaluate(
        self,
        y_pred: Union[Tensor, np.ndarray],
        y_true: Optional[Union[Tensor, np.ndarray]] = None,
        batch: Optional[Batch] = None,
    ) -> Union[float, list[float]]:
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

        y_pred, y_true = self._check_input(y_pred, y_true)

        if isinstance(metric, list):
            return [met(y_pred, y_true).item() for met in metric]
        return metric(y_pred, y_true).item()

    def get_f1(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing binary F1.

        Returns:
            Callable[[Tensor, Tensor], Tensor]: Metric callable taking
            `(y_pred, y_true)`.
        """
        f1 = torchmetrics.F1Score(task="binary")
        return lambda x, y: f1(x, y)

    def get_acc(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing binary accuracy."""
        acc = torchmetrics.Accuracy(task="binary")
        return lambda x, y: acc(x, y)

    def get_spearman(self, index: int) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a spearman correlation callable for the given output index."""
        spearman = torchmetrics.SpearmanCorrCoef()
        return lambda x, y: spearman(x[:,index], y[:,index])

    def get_r2(self, index: int) -> Callable[[Tensor, Tensor], Tensor]:
        """Return an R2 score callable for the given output index."""
        r2 = torchmetrics.R2Score()
        return lambda x, y: r2(x[:,index], y[:,index])

    def get_closed_gap(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing ClosedGap.

        Note: This metric expects `y_true` shaped (N, K) of runtimes or
        costs per algorithm, and `y_pred` shaped (N, K) of scores or
        probabilities used to select the algorithm.
        """
        return lambda y_pred, y_true: self._get_closed_gap(y_pred, y_true)

    def get_chip_design_score(self) -> Callable[[list[Data], list[Data]], Tensor]:
        """Return a callable computing ChipDesignScore."""
        return lambda y_pred, y_true: self._get_chip_design_score(y_pred, y_true)

    def get_weather_mse(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing Weather_MSE."""
        return lambda y_pred, y_true: self._get_weather_mse(y_pred, y_true)

    def get_mis_size(self) -> Callable[[Tensor, Batch], Tensor]:
        """Return a callable computing MisSize.

        Note: This callable expects `(x, batch)` rather than the standard
        `(y_pred, y_true)` signature.
        """
        return lambda x, batch, dec_length=300, num_seeds=1: mis_size(
            mis_decoder(x, batch, dec_length=dec_length, num_seeds=num_seeds)
        )

    def get_max_cut_size(self) -> Callable[[Tensor, Batch], Tensor]:
        """Return a callable computing MaxCutSize.

        Note: This callable expects `(x, batch)` rather than the standard
        `(y_pred, y_true)` signature.
        """
        return lambda x, batch: max_cut_size(max_cut_decoder(x, batch), batch)

    def get_num_colors_used(self) -> Callable[[Tensor, Batch], Tensor]:
        """Return a callable computing NumColorsUsed.

        Note: This callable expects `(x, batch)` rather than the standard
        `(y_pred, y_true)` signature.
        """
        return lambda x, batch, num_seeds=1: num_colors_used(graph_coloring_decoder(x, batch, num_seeds=num_seeds))

    def _get_closed_gap(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        inference_times: Optional[Union[list[Tensor], tuple[Tensor, ...]]] = None,
    ) -> Tensor:

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

    def _get_chip_design_score(self, y_pred: list[Data], y_true: list[Data]) -> Tensor:
        """Compute a chip design equivalence score.

        This method expects `y_pred` and `y_true` to be sequences of
        circuit-like data objects. For each pair it attempts to simulate
        truth-tables using the VectorizedCircuitSimulator class and compares
        outputs. The returned score is in >= 0 with 100 as the score obtained for providing the reference solution.
        """
        if len(y_pred) != len(y_true):
            return torch.tensor(0.)

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

        score = (100.0 * total_score) / N if N > 0 else 0.0
        return torch.tensor(score)

    def _extract_truth_vectors(self, truth_vectors, num_inputs, num_outputs):
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

    def _extract_input_output_counts(self, x: Tensor):
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

    def _equivalence_score(
        self,
        predicted_truth_vectors: np.ndarray,
        original_truth_vectors: np.ndarray,
        num_nodes_generated: int,
        num_nodes_test: int,
    ) -> float:
        if np.array_equal(predicted_truth_vectors, original_truth_vectors):
            # equivalence
            if num_nodes_generated > 0:
                return num_nodes_test / num_nodes_generated
            else:
                return 0.0
        else:
            # No match
            return 0.0

    # TODO I annotated y_true as Tensor, because masked_loss expects a Tensor. However, in the comment it looks like
    #      y_true needs to be a Data object. To me it looks like the implementation is simply incorrect.
    def _get_weather_mse(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
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

    def get_mse(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing mean squared error (averaged per-column)."""
        return lambda y_pred, y_true: self._mse(y_pred, y_true)

    def get_rmse(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing root mean squared error (averaged per-column)."""
        return lambda y_pred, y_true: self._rmse(y_pred, y_true)

    def get_mae(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing mean absolute error (averaged per-column)."""
        return lambda y_pred, y_true: self._mae(y_pred, y_true)

    def get_rse(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a callable computing relative squared error (averaged per-column)."""
        return lambda y_pred, y_true: self._rse(y_pred, y_true)

    def _mse(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_list: list[Tensor] = []

        for i in range(y_true.shape[1]):
            mse_list.append(((y_true[:,i] - y_pred[:,i])**2).mean())

        return sum(mse_list) / len(mse_list)

    def _rmse(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        rmse_list: list[Tensor] = []

        for i in range(y_true.shape[1]):
            rmse_list.append(torch.sqrt(((y_true[:,i] - y_pred[:,i])**2).mean()))

        return sum(rmse_list) / len(rmse_list)

    def _mae(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mae_list: list[Tensor] = []

        for i in range(y_true.shape[1]):
            mae_list.append((torch.abs(y_true[:,i] - y_pred[:,i])).mean())

        return sum(mae_list) / len(mae_list)

    def _rse(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Relative squared error (RSE) averaged over columns.

        For each output dimension $i$:
            $$\mathrm{RSE}_i = \frac{\mathbb{E}[(y_i-\hat{y}_i)^2]}{\mathbb{E}[(y_i-\mathbb{E}[y_i])^2]}$$

        Returns NaN if the variance of `y_true[:, i]` is zero.
        """
        rse_vals: list[Tensor] = []

        for i in range(y_true.shape[1]):
            num = torch.mean((y_true[:, i] - y_pred[:, i]) ** 2)
            denom = torch.var(y_true[:, i], unbiased=False)
            if torch.isclose(denom, torch.tensor(0.0, device=denom.device)):
                rse_vals.append(torch.tensor(float("nan"), device=denom.device))
            else:
                rse_vals.append(num / denom)

        return sum(rse_vals) / len(rse_vals)
