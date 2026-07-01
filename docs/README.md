# Generating Documentation

## Dependencies

Install sphinx and the PyG theme.
We're going to ignore the fact that the PyG theme requires an extremely old version of Sphinx and install a newer version instead.
The second install command will show some dependency conflicts because of that, but that shouldn't be a problem.

```
pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
pip install -U sphinx==9.1.0
```


## Generating HTML Documentation

Make sure the current working directory is `docs/`.
Then run:

1. `rm -r _build` (Linux/macOS) or `rmdir /s /q _build` (Windows)
2. `make html` (Linux) or `make.bat html` (Windows)

The output appears in `_build/html`


## Maintaining API Reference Pages

API reference `.rst` files in `api/` are committed and curated manually.

When the public API changes (new class/module, rename, removal), update these files in the same PR:

1. Add/remove the corresponding `api/*.rst` page.
2. Update `api/index.rst` and the relevant group toctree (`core.rst`, `datasets.rst`, or `graphbench.helpers.rst`).
3. Run `make html` or `make.bat html` to validate the navigation and links.
