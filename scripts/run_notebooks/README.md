# Notebook Runner

Run all of the specified notebooks with mocked sampling in order to check for runtime errors.

## Usage

Run all of the notebooks

```bash
python scripts/run_notebooks/runner.py
```

This is stored in the Makefile as well and be run with the `run_notebooks` target.

```bash
make run_notebooks
```

Specific notebooks can be run by passing them to the `--notebooks` arguments. Use the full path to the notebook. For example:

```bash
python scripts/run_notebooks/runner.py --notebooks docs/source/notebooks/bass/bass_example.ipynb <another-notebook>
```
