#   Copyright 2022 - 2025 The PyMC Labs Developers
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
import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm.preprocessing import (
    MaxAbsScaleChannels,
    MaxAbsScaleTarget,
    StandardizeControls,
    preprocessing_method_X,
)

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
date_data: pd.DatetimeIndex = pd.date_range(
    start="2019-06-01", end="2021-12-31", freq="W-MON"
)

n: int = date_data.size

toy_X = pd.DataFrame(
    data={
        "channel_1": rng.integers(low=0, high=400, size=n),
        "channel_2": rng.integers(low=0, high=50, size=n),
        "control_1": rng.gamma(shape=1000, scale=500, size=n),
        "control_2": rng.gamma(shape=100, scale=5, size=n),
        "other_column_1": rng.integers(low=0, high=100, size=n),
        "other_column_2": rng.normal(loc=0, scale=1, size=n),
    }
)
toy_y = pd.Series(rng.integers(low=0, high=100, size=n))


def test_preprocessing_method():
    f = lambda x: x  # noqa: E731
    f.__doc__ = "bla"
    vf = preprocessing_method_X(f)
    assert getattr(vf, "_tags", {}).get("preprocessing_X", False)
    assert vf.__doc__ == f.__doc__
    assert vf.__name__ == f.__name__

    def f2(x):
        """Bla"""
        return x

    vf = preprocessing_method_X(f2)
    assert getattr(vf, "_tags", {}).get("preprocessing_X", False)
    assert vf.__doc__ == f2.__doc__
    assert vf.__name__ == f2.__name__

    class F:
        @preprocessing_method_X
        def f3(self, x):
            """Bla"""
            return x

    vf = F().f3
    assert getattr(vf, "_tags", {}).get("preprocessing_X", False)
    assert F.f3.__doc__ == vf.__doc__
    assert F.f3.__name__ == vf.__name__
    assert vf.__doc__ == "Bla"
    assert vf.__name__ == "f3"


@pytest.mark.parametrize("to_numpy", [True, False])
def test_max_abs_scale_target(to_numpy: bool):
    obj = MaxAbsScaleTarget()
    data = toy_y.to_numpy() if to_numpy else toy_y
    out = obj.max_abs_scale_target_data(data)
    temp = toy_y
    assert out.min() == temp.min() / temp.max()
    assert out.max() == 1
    # out and temp are both series, therefore the index comparison makes no longer sense


def test_max_abs_scale_channels():
    obj = MaxAbsScaleChannels()
    obj.channel_columns = ["channel_1", "channel_2"]
    out = obj.max_abs_scale_channel_data(toy_X)[obj.channel_columns]
    temp = toy_X[obj.channel_columns]
    assert (out.max(axis=0) == 1).all()
    assert np.allclose(out.min(axis=0), temp.min(axis=0) / temp.max(axis=0))
    pd.testing.assert_index_equal(out.index, toy_X[obj.channel_columns].index)


def test_standardize_controls():
    obj = StandardizeControls()
    obj.control_columns = ["control_1", "control_2"]
    out = obj.standardize_control_data(toy_X)[obj.control_columns]
    assert np.allclose(out.mean(axis=0), 0)
    assert np.allclose(out.std(axis=0), 1, atol=5e-3)
    pd.testing.assert_index_equal(out.index, toy_X[obj.control_columns].index)
