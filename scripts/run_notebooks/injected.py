"""Injected code to the top of each notebook to mock long running code."""

import sys

import numpy as np
import pymc as pm
import xarray as xr


# Disable ipywidgets to avoid nbclient display_id assertion errors.
# When ipywidgets is installed, tqdm uses widget progress bars which cause
# issues when running notebooks with papermill/nbclient.
# Creating a module that raises ImportError forces tqdm to fall back to text mode.
class _BrokenModule:
    """A module placeholder that raises ImportError on attribute access."""

    def __getattr__(self, name):
        raise ImportError("ipywidgets disabled for notebook testing")


sys.modules["ipywidgets"] = _BrokenModule()


def mock_sample(*args, **kwargs):
    random_seed = kwargs.get("random_seed", None)
    model = kwargs.get("model", None)
    samples = 10
    idata = pm.sample_prior_predictive(
        model=model,
        random_seed=random_seed,
        draws=samples,
    )
    idata.add_groups(posterior=idata.prior)

    # Create mock sample stats with diverging data
    if "sample_stats" not in idata:
        n_chains = 1
        n_draws = samples
        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    np.zeros((n_chains, n_draws), dtype=int),
                    dims=("chain", "draw"),
                )
            }
        )
        idata.add_groups(sample_stats=sample_stats)

    del idata.prior
    if "prior_predictive" in idata:
        del idata.prior_predictive
    return idata


pm.sample = mock_sample
pm.HalfFlat = pm.HalfNormal
pm.Flat = pm.Normal
