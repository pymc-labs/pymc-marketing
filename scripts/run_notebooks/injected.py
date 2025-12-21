"""Injected code to the top of each notebook to mock long running code."""

import os

# Disable tqdm progress bars to avoid nbclient display_id assertion errors
# when ipywidgets is installed. Must be set before importing pymc.
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import pymc as pm
import xarray as xr

# Store original functions before mocking
_original_sample_prior_predictive = pm.sample_prior_predictive
_original_sample_posterior_predictive = pm.sample_posterior_predictive


def mock_sample(*args, **kwargs):
    random_seed = kwargs.get("random_seed", None)
    model = kwargs.get("model", None)
    samples = 10
    idata = _original_sample_prior_predictive(
        model=model,
        random_seed=random_seed,
        draws=samples,
        progressbar=False,
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


def mock_sample_prior_predictive(*args, **kwargs):
    """Wrapper to disable progress bar for sample_prior_predictive."""
    kwargs["progressbar"] = False
    return _original_sample_prior_predictive(*args, **kwargs)


def mock_sample_posterior_predictive(*args, **kwargs):
    """Wrapper to disable progress bar for sample_posterior_predictive."""
    kwargs["progressbar"] = False
    return _original_sample_posterior_predictive(*args, **kwargs)


pm.sample = mock_sample
pm.sample_prior_predictive = mock_sample_prior_predictive
pm.sample_posterior_predictive = mock_sample_posterior_predictive
pm.HalfFlat = pm.HalfNormal
pm.Flat = pm.Normal
