import numpy as np
import pandas as pd
import pytest

from pymmmc.mmm.validating import (
    ValidateChannelColumns,
    ValidateControlColumns,
    ValidateDateColumn,
    ValidateTargetColumn,
    validation_method,
)

seed: int = sum(map(ord, "pymmmc"))
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


def test_validation_method():
    f = lambda x: x  # noqa: E731
    f.__doc__ = "bla"
    vf = validation_method(f)
    assert getattr(vf, "_tags", {}).get("validation", False)
    assert vf.__doc__ == f.__doc__
    assert vf.__name__ == f.__name__

    def f2(x):
        """bla"""
        return x

    vf = validation_method(f2)
    assert getattr(vf, "_tags", {}).get("validation", False)
    assert vf.__doc__ == f2.__doc__
    assert vf.__name__ == f2.__name__

    class F:
        @validation_method
        def f3(self, x):
            """bla"""
            return x

    vf = F().f3
    assert getattr(vf, "_tags", {}).get("validation", False)
    assert F.f3.__doc__ == vf.__doc__
    assert F.f3.__name__ == vf.__name__
    assert vf.__doc__ == "bla"
    assert vf.__name__ == "f3"


def test_validate_target():
    obj = ValidateTargetColumn()
    obj.target_column = "y"
    assert obj.validate_target(toy_df) is None

    with pytest.raises(ValueError, match="target y not in data_df"):
        obj.validate_target(toy_df.drop(columns=["y"]))


def test_validate_date_col():
    obj = ValidateDateColumn()
    obj.date_column = "date"
    assert obj.validate_date_col(toy_df) is None
    with pytest.raises(ValueError, match="date_col date not in data_df"):
        obj.validate_date_col(toy_df.drop(columns=["date"]))
    with pytest.raises(ValueError, match="date_col date has repeated values"):
        obj.validate_date_col(pd.concat([toy_df, toy_df], ignore_index=True, axis=0))


def test_channel_columns():
    obj = ValidateChannelColumns()
    obj.channel_columns = ["channel_1", "channel_2"]
    assert obj.validate_channel_columns(toy_df) is None

    with pytest.raises(ValueError, match="channel_columns must be a list or tuple"):
        obj.channel_columns = {}
        obj.validate_channel_columns(toy_df)
    with pytest.raises(ValueError, match="channel_columns must not be empty"):
        obj.channel_columns = []
        obj.validate_channel_columns(toy_df)
    with pytest.raises(
        ValueError,
        match="channel_columns \['out_of_columns'\] not in data_df",  # noqa: W605
    ):
        obj.channel_columns = ["out_of_columns"]
        obj.validate_channel_columns(toy_df)
    with pytest.raises(
        ValueError,
        match="channel_columns \['channel_1', 'channel_1'\] contains duplicates",  # noqa: E501, W605
    ):
        obj.channel_columns = ["channel_1", "channel_1"]
        obj.validate_channel_columns(toy_df)
    with pytest.raises(
        ValueError,
        match="channel_columns \['channel_1'\] contains negative values",  # noqa: E501, W605
    ):
        new_toy_df = toy_df.copy()
        new_toy_df["channel_1"] -= 1e4
        obj.channel_columns = ["channel_1"]
        obj.validate_channel_columns(new_toy_df)


def test_control_columns():
    obj = ValidateControlColumns()
    obj.control_columns = None
    assert obj.validate_control_columns(toy_df) is None
    obj.control_columns = ["control_1", "control_2"]
    assert obj.validate_control_columns(toy_df) is None

    with pytest.raises(
        ValueError, match="control_columns must be None, a list or tuple"
    ):
        obj.control_columns = {}
        obj.validate_control_columns(toy_df)
    with pytest.raises(
        ValueError, match="If control_columns is not None, then it must not be empty"
    ):
        obj.control_columns = []
        obj.validate_control_columns(toy_df)
    with pytest.raises(
        ValueError,
        match="control_columns \['out_of_columns'\] not in data_df",  # noqa: W605
    ):
        obj.control_columns = ["out_of_columns"]
        obj.validate_control_columns(toy_df)
    with pytest.raises(
        ValueError,
        match="control_columns \['control_1', 'control_1'\] contains duplicates",  # noqa: E501, W605
    ):
        obj.control_columns = ["control_1", "control_1"]
        obj.validate_control_columns(toy_df)
