from __future__ import annotations

from contextlib import contextmanager

import pymc as pm
from pymc.testing import mock_sample


@contextmanager
def mock_sampling():
    """Temporarily replace ``pm.sample`` with ``mock_sample`` for speed.

    Also replaces ``HalfFlat`` and ``Flat`` distributions with ``HalfNormal``
    and ``Normal`` respectively to support mock sampling.
    """
    # Store originals
    original_sample = pm.sample
    original_flat = pm.Flat
    original_half_flat = pm.HalfFlat

    # Replace with mock versions
    pm.sample = mock_sample
    pm.Flat = pm.Normal
    pm.HalfFlat = pm.HalfNormal

    try:
        yield
    finally:
        # Restore originals
        pm.sample = original_sample
        pm.Flat = original_flat
        pm.HalfFlat = original_half_flat
