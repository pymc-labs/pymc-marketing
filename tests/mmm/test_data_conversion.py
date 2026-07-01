#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Tests for the data_conversion module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm.data_conversion import to_mmm_dataset


@pytest.fixture
def sample_dataframe():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="W-MON"),
            "x1": rng.uniform(0.1, 1.0, n),
            "x2": rng.uniform(0.1, 1.0, n),
        }
    )


def test_to_mmm_dataset_no_setting_with_copy_warning(sample_dataframe):
    """to_mmm_dataset should not trigger SettingWithCopyWarning on DataFrame slices."""
    df_slice = sample_dataframe[sample_dataframe["x1"] > 0.4]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        to_mmm_dataset(
            df_slice,
            date_column="date",
            channel_columns=["x1", "x2"],
        )

    setting_with_copy = [
        w for w in captured if issubclass(w.category, pd.errors.SettingWithCopyWarning)
    ]
    assert len(setting_with_copy) == 0, (
        f"Got {len(setting_with_copy)} SettingWithCopyWarning(s): "
        f"{[str(w.message) for w in setting_with_copy]}"
    )


def test_to_mmm_dataset_no_setting_with_copy_warning_head(sample_dataframe):
    """to_mmm_dataset should not trigger SettingWithCopyWarning on .head() slices."""
    df_slice = sample_dataframe.head(50)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        to_mmm_dataset(
            df_slice,
            date_column="date",
            channel_columns=["x1", "x2"],
        )

    setting_with_copy = [
        w for w in captured if issubclass(w.category, pd.errors.SettingWithCopyWarning)
    ]
    assert len(setting_with_copy) == 0, (
        f"Got {len(setting_with_copy)} SettingWithCopyWarning(s): "
        f"{[str(w.message) for w in setting_with_copy]}"
    )
