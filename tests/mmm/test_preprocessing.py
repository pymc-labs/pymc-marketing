import numpy as np
import pandas as pd

from pymc_marketing.mmm.preprocessing import (
    MaxAbsScaleChannels,
    MixMaxScaleTarget,
    StandardizeControls,
    preprocessing_method,
)

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
date_data: pd.DatetimeIndex = pd.date_range(
    start="2019-06-01", end="2021-12-31", freq="W-MON"
)

n: int = date_data.size

toy_df = pd.DataFrame(
    data={
        "date": date_data,
        "y": rng.integers(low=0, high=100, size=n),
        "channel_1": rng.integers(low=0, high=400, size=n),
        "channel_2": rng.integers(low=0, high=50, size=n),
        "control_1": rng.gamma(shape=1000, scale=500, size=n),
        "control_2": rng.gamma(shape=100, scale=5, size=n),
        "other_column_1": rng.integers(low=0, high=100, size=n),
        "other_column_2": rng.normal(loc=0, scale=1, size=n),
    }
)


def test_preprocessing_method():
    f = lambda x: x  # noqa: E731
    f.__doc__ = "bla"
    vf = preprocessing_method(f)
    assert getattr(vf, "_tags", {}).get("preprocessing", False)
    assert vf.__doc__ == f.__doc__
    assert vf.__name__ == f.__name__

    def f2(x):
        """bla"""
        return x

    vf = preprocessing_method(f2)
    assert getattr(vf, "_tags", {}).get("preprocessing", False)
    assert vf.__doc__ == f2.__doc__
    assert vf.__name__ == f2.__name__

    class F:
        @preprocessing_method
        def f3(self, x):
            """bla"""
            return x

    vf = F().f3
    assert getattr(vf, "_tags", {}).get("preprocessing", False)
    assert F.f3.__doc__ == vf.__doc__
    assert F.f3.__name__ == vf.__name__
    assert vf.__doc__ == "bla"
    assert vf.__name__ == "f3"


def test_min_max_scale_target():
    obj = MixMaxScaleTarget()
    obj.target_column = "y"
    out = obj.min_max_scale_target_data(toy_df)["y"]
    assert out.min() == 0
    assert out.max() == 1
    pd.testing.assert_index_equal(out.index, toy_df.index)


def test_max_abs_scale_channels():
    obj = MaxAbsScaleChannels()
    obj.channel_columns = ["channel_1", "channel_2"]
    out = obj.max_abs_scale_channel_data(toy_df)[obj.channel_columns]
    temp = toy_df[obj.channel_columns]
    assert (out.max(axis=0) == 1).all()
    assert np.allclose(out.min(axis=0), temp.min(axis=0) / temp.max(axis=0))
    pd.testing.assert_index_equal(out.index, toy_df.index)


def test_standardize_controls():
    obj = StandardizeControls()
    obj.control_columns = ["control_1", "control_2"]
    out = obj.standardize_control_data(toy_df)[obj.control_columns]
    assert np.allclose(out.mean(axis=0), 0)
    assert np.allclose(out.std(axis=0), 1, atol=5e-3)
    pd.testing.assert_index_equal(out.index, toy_df.index)
