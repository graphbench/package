import unittest
from unittest.mock import MagicMock, patch

try:
    from graphbench import Loader
    HAS_GRAPHBENCH = True
except ImportError:
    HAS_GRAPHBENCH = False


class TestGraphBenchLoaderBehavior(unittest.TestCase):

    def setUp(self):
        if not HAS_GRAPHBENCH:
            self.skipTest("GraphBench package not found")

        self.patcher_algoreas = patch("graphbench.datasets.AlgoReasDataset")
        self.patcher_bluesky = patch("graphbench.datasets.BlueSkyDataset")
        self.patcher_chip = patch("graphbench.datasets.ChipDesignDataset")
        self.patcher_co = patch("graphbench.datasets.CODataset")
        self.patcher_ec = patch("graphbench.datasets.ECDataset")
        self.patcher_sat = patch("graphbench.datasets.SATDataset")
        self.patcher_weather = patch("graphbench.datasets.WeatherforecastingDataset")
        self.patcher_split = patch("graphbench._loader.split_dataset")

        self.mock_algoreas = self.patcher_algoreas.start()
        self.mock_bluesky = self.patcher_bluesky.start()
        self.mock_chip = self.patcher_chip.start()
        self.mock_co = self.patcher_co.start()
        self.mock_ec = self.patcher_ec.start()
        self.mock_sat = self.patcher_sat.start()
        self.mock_weather = self.patcher_weather.start()
        self.mock_split = self.patcher_split.start()

    def tearDown(self):
        self.patcher_algoreas.stop()
        self.patcher_bluesky.stop()
        self.patcher_chip.stop()
        self.patcher_co.stop()
        self.patcher_ec.stop()
        self.patcher_sat.stop()
        self.patcher_weather.stop()
        self.patcher_split.stop()

    def test_loader_algoreas_flow(self):
        loader = Loader(root="/tmp", dataset_names="algoreas_flow")
        train_ds = MagicMock(name="train")
        valid_ds = MagicMock(name="valid")
        split_test_ds = MagicMock(name="split_test")
        self.mock_split.return_value = (train_ds, valid_ds, split_test_ds)

        result = loader._loader("algoreas_flow")

        self.assertEqual(set(result.keys()), {"train", "valid", "test"})
        self.assertIs(result["train"], train_ds)
        self.assertIs(result["valid"], valid_ds)
        self.assertIsNot(result["test"], split_test_ds)

        self.mock_split.assert_called_once()
        self.mock_algoreas.assert_any_call(
            root="/tmp",
            name="algoreas_flow_16",
            pre_filter=None,
            pre_transform=None,
            transform=None,
            split="train",
            generate=False,
        )
        self.mock_algoreas.assert_any_call(
            root="/tmp",
            name="algoreas_flow_64",
            pre_filter=None,
            pre_transform=None,
            transform=None,
            split="test",
            generate=False,
        )

    def test_loader_algoreas_sizegen(self):
        loader = Loader(root="/tmp", dataset_names="algoreas_sizegen")
        result = loader._loader("algoreas_sizegen")

        self.assertEqual(set(result.keys()), {"train", "valid", "test"})
        self.assertIsNone(result["train"])
        self.assertIsNone(result["valid"])
        self.mock_algoreas.assert_called_once_with(
            root="/tmp",
            name="algoreas_sizegen",
            pre_filter=None,
            pre_transform=None,
            transform=None,
            split="test",
            generate=False,
        )

    def test_loader_bluesky(self):
        loader = Loader(root="/tmp", dataset_names="bluesky")
        train_ds = MagicMock(name="train")
        valid_ds = MagicMock(name="valid")
        test_ds = MagicMock(name="test")
        self.mock_bluesky.side_effect = [train_ds, valid_ds, test_ds]

        result = loader._loader("bluesky")

        self.assertEqual(result, {"train": train_ds, "valid": valid_ds, "test": test_ds})
        self.assertEqual(self.mock_bluesky.call_count, 3)

    def test_loader_chipdesign(self):
        loader = Loader(root="/tmp", dataset_names="chipdesign")
        train_ds = MagicMock(name="train")
        valid_ds = MagicMock(name="valid")
        test_ds = MagicMock(name="test")
        self.mock_chip.side_effect = [train_ds, valid_ds, test_ds]

        result = loader._loader("chipdesign")

        self.assertEqual(result, {"train": train_ds, "valid": valid_ds, "test": test_ds})
        self.assertEqual(self.mock_chip.call_count, 3)

    def test_loader_weather_uses_split_dataset(self):
        loader = Loader(root="/tmp", dataset_names="weather")
        train_ds = MagicMock(name="train")
        valid_ds = MagicMock(name="valid")
        test_ds = MagicMock(name="test")
        self.mock_split.return_value = (train_ds, valid_ds, test_ds)

        result = loader._loader("weather")

        self.mock_split.assert_called_once()
        args, _ = self.mock_split.call_args
        self.assertEqual(args[1:], (0.8, 0.1, 0.1))
        self.assertEqual(result, {"train": train_ds, "valid": valid_ds, "test": test_ds})
        self.mock_weather.assert_called_once()

    def test_loader_co_uses_split_dataset(self):
        loader = Loader(root="/tmp", dataset_names="co")
        train_ds = MagicMock(name="train")
        valid_ds = MagicMock(name="valid")
        test_ds = MagicMock(name="test")
        self.mock_split.return_value = (train_ds, valid_ds, test_ds)

        result = loader._loader("co")

        self.mock_split.assert_called_once()
        args, _ = self.mock_split.call_args
        self.assertEqual(args[1:], (0.7, 0.15, 0.15))
        self.assertEqual(result, {"train": train_ds, "valid": valid_ds, "test": test_ds})
        self.mock_co.assert_called_once()

    def test_loader_sat_uses_split_dataset(self):
        loader = Loader(
            root="/tmp",
            dataset_names="sat",
            solver="minisat",
            use_satzilla_features=True,
        )
        train_ds = MagicMock(name="train")
        valid_ds = MagicMock(name="valid")
        test_ds = MagicMock(name="test")
        self.mock_split.return_value = (train_ds, valid_ds, test_ds)

        result = loader._loader("sat")

        self.mock_split.assert_called_once()
        args, _ = self.mock_split.call_args
        self.assertEqual(args[1:], (0.8, 0.1, 0.1))
        self.assertEqual(result, {"train": train_ds, "valid": valid_ds, "test": test_ds})
        self.mock_sat.assert_called_once()

    def test_loader_electronic_circuits(self):
        loader = Loader(root="/tmp", dataset_names="electronic_circuits")
        train_ds = MagicMock(name="train")
        valid_ds = MagicMock(name="valid")
        test_ds = MagicMock(name="test")
        self.mock_ec.side_effect = [train_ds, valid_ds, test_ds]

        result = loader._loader("electronic_circuits")

        self.assertEqual(result, {"train": train_ds, "valid": valid_ds, "test": test_ds})
        self.assertEqual(self.mock_ec.call_count, 3)
