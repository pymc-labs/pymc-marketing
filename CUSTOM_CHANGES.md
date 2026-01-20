# Custom Positive Seasonality Implementation

## Overview

This fork adds a custom PositiveSeasonality component to pymc-marketing that guarantees the baseline (intercept + seasonality) stays positive.

## Changes Made

### 1. New Files Created

- `pymc_marketing/mmm/components/positive_seasonality.py` - Core component
- `pymc_marketing/mmm/patches/__init__.py` - Patch module init
- `pymc_marketing/mmm/patches/positive_seasonality_patch.py` - Integration patch

### 2. Implementation Details

**PositiveSeasonality Component:**

- Uses exponential transformation: `seasonality = offset + exp(fourier_sum)`
- Guarantees output is always positive
- Configurable `prior_scale` parameter for controlling seasonal variation
- Based on upstream pymc-marketing v0.17.0

**Integration Patch:**

- `patch_mmm_seasonality()` function for easy MMM integration
- Non-invasive: patches the build_model method
- Configurable prior_scale parameter

### 3. Usage

```python
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.patches import patch_mmm_seasonality

# Create MMM model
mmm = MMM(
    yearly_seasonality=2,
    ...
)

# Apply positive seasonality patch
mmm = patch_mmm_seasonality(mmm, prior_scale=0.05)

# Fit model
mmm.fit(X, y)
```

### 4. Benefits

- ✅ Prevents negative baseline values
- ✅ Maintains compatibility with existing MMM API
- ✅ Configurable seasonal variation
- ✅ Easy to integrate and remove

## Version

Based on: pymc-marketing v0.17.0
Custom version: v0.17.0-tw-1 (Triple Whale Custom Release 1)

## Installation

```bash
pip install git+https://github.com/RamiFisherTW/pymc-marketing.git@v0.17.0-tw-1
```

## Maintenance

- Development branch: `dev-positive-seasonality`
- Production branch: `main`
- Upstream: `https://github.com/pymc-labs/pymc-marketing.git`
