# Notebook Runner

Provides tools for executing Jupyter notebooks in the `docs/source/notebooks/`
directory in two modes:

| Mode | Purpose | Backend |
|---|---|---|
| **Smoke-test** | Fast CI check for runtime errors | `papermill` + mocked PyMC sampling |
| **Real execution** | Full run-through with actual sampling | `jupyter execute` |

---

## Smoke-test (fast, mock mode)

Runs notebooks with `pm.sample` replaced by a fast mock. Use for CI or quick
local checks.

### All notebooks

```bash
# Directly
python scripts/run_notebooks/runner.py

# Makefile commands
make run_notebooks        # all notebooks
make run_notebooks_mmm    # MMM only
make run_notebooks_other  # non-MMM only
```

### Specific notebook(s)

Pass relative paths from the repo root:

```bash
python scripts/run_notebooks/runner.py \
    --notebooks docs/source/notebooks/bass/bass_example.ipynb
```

### Directory / index range

```bash
# Run all notebooks in a directory
python scripts/run_notebooks/runner.py --notebooks mmm

# Run a slice (3rd through 5th notebook in mmm/)
python scripts/run_notebooks/runner.py --notebooks mmm --start-idx 2 --end-idx 5

# Exclude directories
python scripts/run_notebooks/runner.py --exclude-dirs mmm clv

# Run in parallel
python scripts/run_notebooks/runner.py --parallel
```

---

## Real execution (headless, full sampling)

Uses `jupyter execute` to run every cell for real with no mocking.
Useful for validating notebook output during migration or release.

### Single notebook

```bash
uv run jupyter execute docs/source/notebooks/mmm/mmm_example.ipynb
```

Save executed results (outputs including plots) back to the notebook:

```bash
uv run jupyter execute docs/source/notebooks/mmm/mmm_example.ipynb --inplace
```

### All notebooks in a directory

```bash
for nb in docs/source/notebooks/mmm/*.ipynb; do
    uv run jupyter execute "$nb" --inplace
done
```

### All MMM notebooks (alternative)

```bash
find docs/source/notebooks/mmm -name '*.ipynb' -exec uv run jupyter execute {} --inplace \;
```

### Clearing stale outputs

Before re-executing, clear any outputs that may contain stale or invalid
metadata:

```bash
uv run jupyter nbconvert --clear-output docs/source/notebooks/mmm/mmm_quickstart.ipynb
```
