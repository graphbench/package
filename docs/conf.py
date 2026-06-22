# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
from copy import deepcopy

from docutils import nodes

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
    "sphinx.ext.mathjax",      # Render LaTeX math
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
napoleon_custom_sections = ["Overview", "Helpers", "Graph Attributes", "List of Available Datasets", "Splits", "Usage Notes"]


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


def _inline_rendered_return_type_fields(app, doctree, docname):
    """
    Rewrite Sphinx field lists so return types appear inline with return descriptions.

    Why this hook exists:
    - With ``autodoc_typehints = "both"``, Sphinx renders two separate fields for return information:
      1) ``RETURNS:`` (the textual description)
      2) ``RETURN TYPE:`` (the annotation/type).
      This visually takes up a lot of space.
    - We instead want the compact format: ``RETURNS: <type> - <description>``.

    What this hook does:
    - Runs at ``doctree-resolved`` after autodoc and napoleon have built the final docutils tree for a page.
    - For each field-list block, if both ``Returns`` and ``Return type`` are present, it moves the rendered return-type
      nodes to the start of the ``Returns`` body, adds a separator, and removes the standalone ``Return type`` field.
    - We copy rendered nodes (not plain text) so type cross-references continue to work, e.g. links to ``Tensor`` docs
      remain clickable.
    """

    def _get_field_name_and_body(field):
        # A docutils field consists of:
        # - field_name: label such as "Returns" or "Return type"
        # - field_body: the content associated with that label
        field_name = None
        field_body = None
        for child in field.children:
            if isinstance(child, nodes.field_name):
                field_name = child
            elif isinstance(child, nodes.field_body):
                field_body = child
        return field_name, field_body

    # Sphinx renders Args/Returns-style sections as docutils field_list blocks.
    for field_list in doctree.findall(nodes.field_list):
        returns_field = None
        return_type_field = None

        # Find sibling fields named "Returns" and "Return type" in this field list.
        for field in [child for child in field_list.children if isinstance(child, nodes.field)]:
            field_name, _ = _get_field_name_and_body(field)
            if field_name is None:
                continue

            normalized_name = field_name.astext().strip().lower()
            if normalized_name == "returns":
                returns_field = field
            elif normalized_name == "return type":
                return_type_field = field

        if returns_field is None or return_type_field is None:
            # This list has nothing to rewrite.
            continue

        _, returns_body = _get_field_name_and_body(returns_field)
        _, return_type_body = _get_field_name_and_body(return_type_field)
        if returns_body is None or return_type_body is None:
            continue

        # Return type content is usually wrapped in a paragraph node.
        return_type_para = next(
            (child for child in return_type_body.children if isinstance(child, nodes.paragraph)),
            None,
        )
        if return_type_para is not None:
            return_type_nodes = [deepcopy(node) for node in return_type_para.children]
        else:
            return_type_nodes = [deepcopy(node) for node in return_type_body.children]

        if not return_type_nodes:
            # Empty Return type field: remove it and continue.
            field_list.remove(return_type_field)
            continue

        # We prepend return type nodes to the first paragraph in Returns.
        first_paragraph = next(
            (child for child in returns_body.children if isinstance(child, nodes.paragraph)),
            None,
        )
        if first_paragraph is None:
            first_paragraph = nodes.paragraph()
            returns_body.insert(0, first_paragraph)

        # Rebuild as: <type nodes> – <existing description nodes>
        description_nodes = list(first_paragraph.children)
        first_paragraph.clear()
        first_paragraph.extend(return_type_nodes)
        if description_nodes:
            first_paragraph += nodes.Text(" \u2013 ")
            first_paragraph.extend(description_nodes)

        # Remove the standalone field because its content is now inlined.
        field_list.remove(return_type_field)


def setup(app):
    app.connect("doctree-resolved", _inline_rendered_return_type_fields)
