# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

# Add docs/ and project root to sys.path before importing local helpers.
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, DOCS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from _return_type_hook import _inline_rendered_return_type_fields



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
    "sphinx.ext.mathjax",      # Render LaTeX math
    "sphinx.ext.intersphinx",  # Allows linking to external documentation
]

exclude_patterns = ["_build"]


def setup(app):
    # Connect custom hook that changes how return types are rendered
    app.connect("doctree-resolved", _inline_rendered_return_type_fields)



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
napoleon_custom_sections = ["Overview", "Helpers", "Graph Attributes", "Targets", "List of Available Datasets", "Splits", "Usage Notes"]


# -- MathJax configuration ---------------------------------------------------
# Allow inline `$...$` TeX delimiters for math rendering in docstrings
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}
# Explicit MathJax CDN path (MathJax v3) to guarantee the JS is injected
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"



# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/main", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}
