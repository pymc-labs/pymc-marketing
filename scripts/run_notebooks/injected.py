"""Injected code to the top of each notebook to mock long running code."""

import numpy as np
import pymc as pm
import xarray as xr
from pymc.testing import mock_sample as pymc_mock_sample


def mock_sample(*args, **kwargs):
    """Wrapper around pymc.testing.mock_sample that adds sample_stats group.

    This wrapper uses pymc.testing.mock_sample as the base implementation
    and adds the sample_stats group with mock diverging data for compatibility
    with notebooks that check for divergences.
    """
    # Use pymc.testing.mock_sample as the base implementation
    idata = pymc_mock_sample(*args, **kwargs)

    # Create mock sample stats with diverging data for notebook compatibility
    if "sample_stats" not in idata:
        # Get the number of chains and draws from the posterior group
        n_chains = idata.posterior.sizes["chain"]
        n_draws = idata.posterior.sizes["draw"]
        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    np.zeros((n_chains, n_draws), dtype=bool),
                    dims=("chain", "draw"),
                )
            }
        )
        idata.add_groups(sample_stats=sample_stats)

    return idata


pm.sample = mock_sample
pm.HalfFlat = pm.HalfNormal
pm.Flat = pm.Normal
