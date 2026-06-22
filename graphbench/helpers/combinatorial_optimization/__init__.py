"""
Helper functions for unsupervised combinatorial optimization (CO).

For each CO problem, we provide:

- An unsupervised loss function that can be used to train a model without ground-truth solutions,
- A decoder that converts the model's soft output to a discrete solution for evaluation at test time,
- A metric that computes the objective value of given solutions, and
- A validator that checks whether a given solution meets the constraints of the CO problem.

In general, validating the output of a decoder is not necessary, since the decoders already guarantee valid outputs.

See :class:`graphbench.datasets.CODataset` for general information on CO.

Quick reference:

.. list-table::
   :header-rows: 1

   * - Function
     - MIS
     - Max-Cut
     - Graph Coloring
   * - Loss
     - :func:`~graphbench.helpers.combinatorial_optimization.mis_loss`
     - :func:`~graphbench.helpers.combinatorial_optimization.max_cut_loss`
     - :func:`~graphbench.helpers.combinatorial_optimization.graph_coloring_loss`
   * - Decoder
     - :func:`~graphbench.helpers.combinatorial_optimization.mis_decoder`
     - :func:`~graphbench.helpers.combinatorial_optimization.max_cut_decoder`
     - :func:`~graphbench.helpers.combinatorial_optimization.graph_coloring_decoder`
   * - Metric
     - :func:`~graphbench.helpers.combinatorial_optimization.mis_size`
     - :func:`~graphbench.helpers.combinatorial_optimization.max_cut_size`
     - :func:`~graphbench.helpers.combinatorial_optimization.num_colors_used`
   * - Validator
     - :func:`~graphbench.helpers.combinatorial_optimization.validate_mis_solution`
     - :func:`~graphbench.helpers.combinatorial_optimization.validate_max_cut_solution`
     - :func:`~graphbench.helpers.combinatorial_optimization.validate_chrom_solution`
"""

from ._decoders import graph_coloring_decoder, max_cut_decoder, mis_decoder
from ._metrics import max_cut_size, mis_size, num_colors_used
from ._unsupervised_losses import graph_coloring_loss, max_cut_loss, mis_loss
from ._validate_solution import validate_chrom_solution, validate_max_cut_solution, validate_mis_solution


__all__ = [
    "graph_coloring_decoder",
    "max_cut_decoder",
    "mis_decoder",
    "max_cut_size",
    "mis_size",
    "num_colors_used",
    "graph_coloring_loss",
    "max_cut_loss",
    "mis_loss",
    "validate_chrom_solution",
    "validate_max_cut_solution",
    "validate_mis_solution",
]
