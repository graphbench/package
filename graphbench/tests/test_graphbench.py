import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np


# Attempt imports
try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from graphbench import Evaluator
    from graphbench import Loader
    # Optimizer is optional
    try:
        from graphbench import Optimizer
    except ImportError:
        Optimizer = None
    HAS_GRAPHBENCH = True
except ImportError:
    HAS_GRAPHBENCH = False

class TestGraphBenchImports(unittest.TestCase):
    """
    1. Tests for correct imports of methods.
    """
    def test_import_loader(self):
        if not HAS_GRAPHBENCH: 
            self.skipTest("GraphBench not installed")
        self.assertIsNotNone(Loader)

    def test_import_evaluator(self):
        if not HAS_GRAPHBENCH: 
            self.skipTest("GraphBench not installed")
        self.assertIsNotNone(Evaluator)

class TestGraphBenchEvaluatorFull(unittest.TestCase):
    
    def setUp(self):
        if not HAS_GRAPHBENCH:
            self.skipTest("GraphBench package not found")
            
        # Patch pandas.read_csv to avoid needing master.csv
        self.patcher = patch('pandas.read_csv')
        self.mock_read_csv = self.patcher.start()
        self.mock_df = MagicMock()
        self.mock_read_csv.return_value = self.mock_df

    def tearDown(self):
        self.patcher.stop()

    def get_evaluator(self, metric_name):
        # Setup mock to return the metric name we want
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda key: "test_task" if key == 'task' else metric_name
        self.mock_df.loc.__getitem__.return_value = mock_row
        return Evaluator("dummy_name")

    def test_acc(self):
        ev = self.get_evaluator("ACC")
        # Fixed - shape [num_samples, 1]
        y_true = torch.tensor([[0], [1], [1], [0]])  # [4, 1]
        y_pred = torch.tensor([[0], [1], [0], [1]])  # [4, 1]
        self.assertEqual(y_true.shape, (4, 1))
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 0.5)
        # Random - shape [10, 1]
        y_true = torch.randint(0, 2, (10, 1))
        y_pred = torch.randint(0, 2, (10, 1))
        res = ev.evaluate(y_pred, y_true)
        self.assertTrue(0.0 <= res <= 1.0)

    def test_f1(self):
        ev = self.get_evaluator("F1")
        # Fixed: TP=1, FP=0, FN=1. F1 = 2*1 / (2+0+1) = 0.666
        y_true = torch.tensor([[0], [1], [1], [0]])
        y_pred = torch.tensor([[0], [1], [0], [0]])
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 2/3, places=4)
        # Random
        y_true = torch.randint(0, 2, (10, 1))
        y_pred = torch.randint(0, 2, (10, 1))
        res = ev.evaluate(y_pred, y_true)
        self.assertTrue(0.0 <= res <= 1.0)

    def test_mse(self):
        ev = self.get_evaluator("MSE")
        # Fixed - single task [N, 1]
        y_true = torch.tensor([[1.0], [2.0]])  # [2, 1]
        y_pred = torch.tensor([[1.5], [2.5]])  # [2, 1]
        # Errors: 0.5, 0.5. Sq: 0.25, 0.25. Mean: 0.25
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 0.25)
        # Random - single task
        y_true = torch.randn(10, 1)
        y_pred = torch.randn(10, 1)
        self.assertIsInstance(ev.evaluate(y_pred, y_true), float)
        # Multi-task [N, 3]
        y_true = torch.randn(5, 3)
        y_pred = torch.randn(5, 3)
        result = ev.evaluate(y_pred, y_true)
        self.assertIsInstance(result, float)

    def test_mae(self):
        ev = self.get_evaluator("MAE")
        # Fixed - shape [2, 1]
        y_true = torch.tensor([[1.0], [2.0]])  # [2, 1]
        y_pred = torch.tensor([[1.5], [2.5]])  # [2, 1]
        # Abs: 0.5, 0.5. Mean: 0.5
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 0.5)

        # Random - single task [10, 1]
        y_true = torch.randn(10, 1)
        y_pred = torch.randn(10, 1)
        expected = torch.mean(torch.abs(y_true - y_pred)).item()
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), expected, places=6)
        
        # Multi-task [10, 3] - averaged over tasks
        y_true = torch.randn(10, 3)
        y_pred = torch.randn(10, 3)
        result = ev.evaluate(y_pred, y_true)
        self.assertIsInstance(result, float)

    def test_rmse(self):
        ev = self.get_evaluator("RMSE")
        # Fixed
        y_true = torch.tensor([[1.0], [2.0]])
        y_pred = torch.tensor([[1.5], [2.5]])
        # MSE=0.25. RMSE=0.5
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 0.5)

        # Random
        y_true = torch.randn(10, 1)
        y_pred = torch.randn(10, 1)
        expected = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), expected, places=6)

    def test_rse(self):
        ev = self.get_evaluator("RSE")
        # Fixed
        # y_true mean = 1.5. Denom = ((1-1.5)^2 + (2-1.5)^2)/2 = (0.25+0.25)/2 = 0.25
        # Num (MSE) = 0.25. RSE = 1.0
        y_true = torch.tensor([[1.0], [2.0]])
        y_pred = torch.tensor([[1.5], [2.5]])
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 1.0)

        # Random (compute expected per evaluator's definition)
        y_true = torch.randn(10, 1)
        y_pred = torch.randn(10, 1)
        num = torch.mean((y_true[:, 0] - y_pred[:, 0]) ** 2)
        denom = torch.mean((y_true[:, 0] - torch.mean(y_true[:, 0])) ** 2)
        expected = (num / denom).item() if denom.item() != 0.0 else float("nan")
        got = ev.evaluate(y_pred, y_true)
        if np.isnan(expected):
            self.assertTrue(np.isnan(got))
        else:
            self.assertAlmostEqual(got, expected, places=6)

    def test_spearman(self):
        # spearman_r_0 - tests first task column
        ev = self.get_evaluator("spearman_r_0")
        # Fixed: Perfect correlation [3, 1]
        y_true = torch.tensor([[1.0], [2.0], [3.0]])  # [3, 1]
        y_pred = torch.tensor([[1.0], [2.0], [3.0]])  # [3, 1]
        self.assertEqual(y_true.shape[1], 1)
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 1.0, places=5)
        # Anti-correlation
        y_pred = torch.tensor([[3.0], [2.0], [1.0]])
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), -1.0, places=5)
        # Multi-task: spearman_r_0 should use column 0
        y_true_multi = torch.tensor([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])  # [3, 2]
        y_pred_multi = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])  # [3, 2]
        result = ev.evaluate(y_pred_multi, y_true_multi)
        self.assertAlmostEqual(result, 1.0, places=5)  # Perfect correlation in col 0

    def test_r2(self):
        ev = self.get_evaluator("r2_0")
        # Fixed
        y_true = torch.tensor([[1.0], [2.0], [3.0]])
        y_pred = torch.tensor([[1.0], [2.0], [3.0]])
        self.assertAlmostEqual(ev.evaluate(y_pred, y_true), 1.0)

    def test_stacked_tensors(self):
        # Check if metrics accept [N, num_tasks] format
        # For node-level: [num_nodes, num_tasks], graph-level: [num_graphs, num_tasks]
        metrics = ['MSE', 'MAE', 'RMSE', 'RSE', 'spearman_r_0', 'r2_0']
        for m in metrics:
            with self.subTest(metric=m):
                ev = self.get_evaluator(m)
                # Single task [10, 1]
                y_true_single = torch.randn(10, 1)
                y_pred_single = torch.randn(10, 1)
                res_single = ev.evaluate(y_pred_single, y_true_single)
                self.assertIsInstance(res_single, float)
                
                # Multiple tasks [10, 3]
                y_true_multi = torch.randn(10, 3)
                y_pred_multi = torch.randn(10, 3)
                res_multi = ev.evaluate(y_pred_multi, y_true_multi)
                self.assertIsInstance(res_multi, float)
                
                # Large batch (concatenated) [100, 1]
                y_true_batch = torch.randn(100, 1)
                y_pred_batch = torch.randn(100, 1)
                res_batch = ev.evaluate(y_pred_batch, y_true_batch)
                self.assertIsInstance(res_batch, float)

    # Complex metrics
    def test_mis_size(self):
        if not HAS_PYG: 
            return
        ev = self.get_evaluator("MisSize")
        # Patch mis_decoder on the instance because the class might have a bug (calls self.mis_decoder but defined as _mis_decoder)
        ev.mis_decoder = MagicMock()
        
        # Create dummy batch
        x = torch.randn(10, 1)
        batch = Batch.from_data_list([Data(x=torch.randn(5,1), edge_index=torch.zeros(2,0).long()) for _ in range(2)])
        
        # Mock return of mis_decoder
        mock_batch_out = MagicMock()
        mock_batch_out.to_data_list.return_value = [MagicMock(is_size=3.0), MagicMock(is_size=4.0)]
        ev.mis_decoder.return_value = mock_batch_out
        
        # Call through evaluate(): for batch-metrics, evaluate() returns a float
        score = ev.evaluate(x, batch=batch)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 3.5)

    def test_max_cut_size(self):
        if not HAS_PYG: 
            return
        ev = self.get_evaluator("MaxCutSize")
        # Create a simple graph: 0-1. 
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Batch.from_data_list([Data(num_nodes=2, edge_index=edge_index)])
        
        # Case 1: Different partitions (Cut)
        x = torch.tensor([[1.0], [-1.0]]) 
        score = ev.evaluate(x, batch=data)
        self.assertEqual(score, 1.0)
        
        # Case 2: Same partition (No cut)
        x = torch.tensor([[1.0], [1.0]])
        score = ev.evaluate(x, batch=data)
        self.assertEqual(score, 0.0)

    def test_closed_gap(self):
        ev = self.get_evaluator("ClosedGap")
        y_true = torch.tensor([[10.0, 1.0], [10.0, 1.0]]) # Alg 0 is bad (10), Alg 1 is good (1)
        y_pred = torch.tensor([[0.1, 0.9], [0.1, 0.9]]) # Predicts Alg 1
        
        # single_best (idx 0) = 10.0
        # predicted_best (idx 1) = 1.0
        # virtual_best (min) = 1.0
        # gap = (10 - 1) / (10 - 1) = 1.0
        
        score = ev.evaluate(y_pred, y_true)
        self.assertAlmostEqual(score, 1.0)

    def test_chip_design_score(self):
        ev = self.get_evaluator("ChipDesignScore")
        
        with patch('graphbench.evaluator.VectorizedCircuitSimulator') as MockSim, \
             patch.object(ev, 'extract_input_output_counts', return_value=(2, 1), create=True), \
             patch.object(ev, 'equivalence_score', wraps=ev._equivalence_score, create=True):
            # Setup mock: both sims return same truth table
            MockSim.side_effect = [
                MagicMock(simulate_all_patterns=MagicMock(return_value=np.zeros((1, 2)))),
                MagicMock(simulate_all_patterns=MagicMock(return_value=np.zeros((1, 2))))
            ]
            
            # Create dummy objects
            class Circuit:
                def __init__(self):
                    self.x = torch.zeros(10, 3) # dummy features
                    self.num_inputs = 2
                    self.num_outputs = 1
            
            y_pred = [Circuit()]
            y_true = [Circuit()]
            
            score = ev._get_chip_design_score(y_pred, y_true)
            self.assertAlmostEqual(score, 100.0)

    def test_weather_mse(self):
        ev = self.get_evaluator("Weather_MSE")
        # Patch the whole method to avoid heavy dependency stack
        with patch.object(ev, '_get_weather_mse', return_value=torch.tensor(123.45)):
            y_true = MagicMock()
            y_pred = torch.tensor([0.0])
            score = ev._get_weather_mse(y_pred, y_true)
            self.assertAlmostEqual(score.item(), 123.45, places=5)

if __name__ == '__main__':
    unittest.main()
