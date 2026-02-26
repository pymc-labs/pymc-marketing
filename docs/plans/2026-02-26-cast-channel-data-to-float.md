# Cast channel_data to float Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Always cast `channel_data` to float at model construction time so the budget optimizer, incrementality, and sensitivity analysis work correctly with integer input data.

**Architecture:** A single guard in `MMM._generate_and_preprocess_model_data()` casts the `_channel` xarray DataArray to float before any downstream consumer sees it. A new test verifies the cast and that the model builds correctly with int inputs.

**Tech Stack:** Python, xarray, numpy, PyMC, pytest

---

### Task 1: Write the failing test

**Files:**
- Test: `tests/mmm/test_multidimensional.py`

**Step 1: Write the failing test**

Add the following test after the existing `test_set_xarray_data_preserves_dtypes` test (around line 2933):

```python
def test_integer_channel_data_is_cast_to_float(multi_dim_data):
    """Integer channel data is cast to float at construction time (GH #2340)."""
    X, y = multi_dim_data

    # Verify the fixture actually produces integer channel columns
    assert X["channel_1"].dtype.kind == "i", "Fixture should produce int channel data"

    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        dims=("country",),
    )
    mmm.build_model(X, y)

    # xarray dataset should have float channel data
    assert np.issubdtype(mmm.xarray_dataset["_channel"].dtype, np.floating)

    # The PyTensor pm.Data node should also be float
    channel_data_var = mmm.model["channel_data"]
    assert np.issubdtype(np.dtype(channel_data_var.type.dtype), np.floating)
```

**Step 2: Run the test to verify it fails**

Run: `conda run -n pymc-marketing-dev pytest tests/mmm/test_multidimensional.py::test_integer_channel_data_is_cast_to_float -v`

Expected: FAIL on the `np.issubdtype(mmm.xarray_dataset["_channel"].dtype, np.floating)` assertion, because the cast hasn't been implemented yet.

**Step 3: Commit the failing test**

```bash
git add tests/mmm/test_multidimensional.py
git commit -m "test: add failing test for int channel_data cast to float (#2340)"
```

---

### Task 2: Implement the cast

**Files:**
- Modify: `pymc_marketing/mmm/multidimensional.py:1418` (in `_generate_and_preprocess_model_data`)

**Step 1: Add the cast**

In `_generate_and_preprocess_model_data()`, immediately after line 1418 (`self.xarray_dataset = xr.merge(dataarrays).fillna(0)`), add:

```python
        if not np.issubdtype(self.xarray_dataset["_channel"].dtype, np.floating):
            self.xarray_dataset["_channel"] = self.xarray_dataset["_channel"].astype(
                float
            )
```

**Step 2: Run the test to verify it passes**

Run: `conda run -n pymc-marketing-dev pytest tests/mmm/test_multidimensional.py::test_integer_channel_data_is_cast_to_float -v`

Expected: PASS

**Step 3: Run the full multidimensional test suite to check for regressions**

Run: `conda run -n pymc-marketing-dev pytest tests/mmm/test_multidimensional.py -x -q`

Expected: All tests pass. (Note: the existing `multi_dim_data` fixture already uses `np.random.randint` so existing tests exercise the cast path.)

**Step 4: Run pre-commit**

Run: `pre-commit run --files pymc_marketing/mmm/multidimensional.py tests/mmm/test_multidimensional.py`

Expected: All checks pass.

**Step 5: Commit**

```bash
git add pymc_marketing/mmm/multidimensional.py tests/mmm/test_multidimensional.py
git commit -m "fix: cast channel_data to float at construction time (#2340)"
```
