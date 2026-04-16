.. GraphBench documentation master file, created by
   sphinx-quickstart on Mon Feb  9 16:33:40 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



GraphBench: Next-generation graph learning benchmarking
=======================================================

We present *GraphBench*, a comprehensive graph learning benchmark across domains and prediction regimes.
GraphBench standardizes evaluation with consistent splits, metrics, and out-of-distribution checks, and includes a
unified hyperparameter tuning framework.
We also provide strong baselines with state-of-the-art message-passing and graph transformer models to get you started.



Code Example
------------


.. code:: python

   import graphbench

   model = # Your torch model
   dataset_name = # Name of the task or list of tasks
   pre_filter = # PyTorch Geometric filter matrix
   pre_transform = # PyTorch Geometric-like transform during loading
   transform = # PyTorch Geometric-like transform at computation time

   # Setting up the components of graphbench
   evaluator = graphbench.Evaluator(dataset_name)
   optimizer = graphbench.Optimizer(optimization_args, training_method)
   loader = graphbench.Loader(dataset_name, pre_filter, pre_transform, transform)

   # Load a GraphBench dataset and get splits
   dataset = loader.load()

   # Optimize your model
   opt_model = optimizer.optimize()

   # Use graphbench evaluator with targets y_true and predictions y_pred
   results = evaluator.evaluate(y_true, y_pred)




Further Information
-------------------

* `Website <https://graphbench.github.io/website/>`__
* `Information on the datasets <https://graphbench.github.io/website/datasets.html>`__
* `Paper <https://arxiv.org/abs/2512.04475>`__



.. toctree::
   :maxdepth: 3
   :caption: Contents

   _api/graphbench.rst
