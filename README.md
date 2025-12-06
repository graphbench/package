# GraphBench: Next generation graph learning benchmarking

This is the package associated with the paper [GraphBench: Next generation graph learning benchmarking](https://arxiv.org/abs/2512.04475). 

It contains the code and tools necessary to load the datasets associated with the benchmark. 
GraphBench is a collection of benchmarking datasets across domains and tasks obtained from real world and synthetic applications. 

## Features

GraphBench comes as a Python package with the following features: 
|        |           |                             
| :------- | :------ |
| Data Loading           | Efficiently loads graph datasets for benchmarking and experimentation across all domains and tasks.       |
| Metric Evaluation      | Supports a wide range of evaluation metrics for graph learning tasks.        |
| Automated Model Tuning | Integrates SMAC3 for automatic hyperparameter optimization of user models.   |

## Installation

GraphBench can be easily installed using the python package manager pip:
```pip install graphbench-lib```

Please make sure to update the installation of GraphBench before running the benchmark for best results. 
Alternatively one can also install from source:

```
git clone https://github.com/graphbench/package
cd package
pip install -e . 
```

## Usage

The package can be easily used to get selected datasets from the GraphBench tasks:

```
import graphbench-lib as graphbench
Loader = graphbench.loader.Loader(root, dataset_name)
datasets = Loader.load()
```

Furthermore, standardized evaluation metrics can be obtained using the following methods:

```
Evaluator = graphbench.evaluator.Evaluator(metric_name)
metric_results = Evaluator.evaluate()
```

For a full list of the datasets, see the accompanying website (https://graphbench.io) or the ```datasets.csv``` file. 

## Citing GraphBench:

If you use GraphBench or GraphBench datasets in your work please cite our paper:
```
@article{GraphBench,
title={GraphBench: Next-generation graph learning benchmarking}, 
author={Timo Stoll and Chendi Qian and Ben Finkelshtein and Ali Parviz and Darius Weber and Fabrizio Frasca and Hadar Shavit and Antoine Siraudin and Arman Mielke and Marie Anastacio and Erik Müller and Maya Bechler-Speicher and Michael Bronstein and Mikhail Galkin and Holger Hoos and Mathias Niepert and Bryan Perozzi and Jan Tönshoff and Christopher Morris},
year={2025},
journal={arXiv preprint arXiv:2512.04475}
}
```