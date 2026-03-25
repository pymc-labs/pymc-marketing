"""Injected code to the top of each notebook to mock long running code."""

from functools import partial

import numpy as np
import pymc as pm
import pymc.testing


def mock_diverging(size):
    return np.zeros(size, dtype=int)


pm.sample = partial(
    pymc.testing.mock_sample,
    sample_stats={"diverging": mock_diverging},
)
pm.HalfFlat = pm.HalfNormal
pm.Flat = pm.Normal
