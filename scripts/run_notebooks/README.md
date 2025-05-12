# Notebook Runner

Run all of the specified notebooks with mocked sampling in order to check for runtime errors.

## Usage

### All notebook

Use either the script directly or the Makefile command.

```bash
# Directly
python scripts/run_notebooks/runner.py

# Makefile command
make run_notebooks
```

### Specific notebook(s)

Specific notebook(s) can be run by passing them to the `--notebooks` arguments. Use the relative path to the notebook. For example:

```bash
python scripts/run_notebooks/runner.py --notebooks docs/source/notebooks/bass/bass_example.ipynb <another-notebook>
```
