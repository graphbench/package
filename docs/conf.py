# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

# Add the project root to sys.path so that Sphinx can find the modules
sys.path.insert(0, os.path.abspath(".."))



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GraphBench"
author = "Timo Stoll, Chendi Qian, Ben Finkelshtein, Ali Parviz, Darius Weber, Fabrizio Frasca, Hadar Shavit, Antoine Siraudin, Arman Mielke, Marie Anastacio, Erik Müller, Maya Bechler-Speicher, Michael Bronstein, Mikhail Galkin, Holger Hoos, Mathias Niepert, Bryan Perozzi, Jan Tönshoff, Christopher Morris"
copyright = "2026, " + author
version = "0.1.2.4"



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",      # Required to read Python docstrings and autogenerate .rst files
    "sphinx.ext.napoleon",     # Allows Sphinx to parse Google-style docstrings
    "sphinx.ext.viewcode",     # Adds links to the source code
    "sphinx.ext.intersphinx",  # Allows linking to external documentation
]

exclude_patterns = ["_build"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pyg_sphinx_theme"  # Use the PyTorch Geometric theme...
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # ...with some adjustments
html_logo = ("_static/GraphBench_logo_text_black.svg")
html_favicon = ("_static/GraphBench_logo.svg")



# -- Autodoc configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Mock imports so we don't have to install all dependencies to build the docs
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torchmetrics",
    "numpy",
    "pandas",
    "smac",
    "requests",
    "networkx",
    "tqdm",
    "sklearn",
    "ConfigSpace",
]

# Show type hints in both the signature and the description
autodoc_typehints = "both"

# Display the signature of the __init__ method separately from the class description
autodoc_class_signature = "separated"



# -- Napoleon configuration --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

# This project uses Google-style docstrings only
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# These names will also be recognized as section headers in Google-style docstrings,
# in addition to the default ones like "Args", "Returns", etc.
napoleon_custom_sections = ["Overview", "Graph Attributes", "List of Available Datasets", "Usage Notes"]



# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/main", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}
