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
]

exclude_patterns = ["_build"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Use the PyTorch Geometric theme
html_theme = "pyg_sphinx_theme"
html_static_path = ["_static"]
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

# Include both class docstring and __init__() docstring
autoclass_content = "both"

# Show type hints in both the signature and the description
autodoc_typehints = "both"



# -- Napoleon configuration --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

# This project uses Google-style docstrings only
napoleon_numpy_docstring = False
