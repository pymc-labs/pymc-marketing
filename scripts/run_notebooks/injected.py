"""Injected code to the top of each notebook to mock long running code."""

import numpy as np
import pymc as pm
import xarray as xr


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
