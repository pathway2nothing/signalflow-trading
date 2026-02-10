# Tutorials

Interactive Jupyter notebooks demonstrating SignalFlow capabilities.

## Adding Notebooks

To add a notebook to documentation:

1. Place your `.ipynb` file in `docs/notebooks/`
2. Add it to `mkdocs.yml` nav:

```yaml
nav:
  - Tutorials:
    - notebooks/index.md
    - My Tutorial: notebooks/my_tutorial.ipynb
```

3. Run `mkdocs serve` to preview

All cell outputs (charts, tables, images) will be rendered automatically.

## Running Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Launch Jupyter
jupyter lab
```
