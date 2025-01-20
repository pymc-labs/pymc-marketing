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

from pymc_marketing.mmm.validating import (
    ValidateChannelColumns,
    ValidateControlColumns,
    ValidateDateColumn,
    ValidateTargetColumn,
    validation_method_X,
)

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
date_data: pd.DatetimeIndex = pd.date_range(
    start="2019-06-01", end="2021-12-31", freq="W-MON"
)

n: int = date_data.size

toy_X = pd.DataFrame(
    data={
        "date": date_data,
        "channel_1": rng.integers(low=0, high=400, size=n),
        "channel_2": rng.integers(low=0, high=50, size=n),
        "control_1": rng.gamma(shape=1000, scale=500, size=n),
        "control_2": rng.gamma(shape=100, scale=5, size=n),
        "other_column_1": rng.integers(low=0, high=100, size=n),
        "other_column_2": rng.normal(loc=0, scale=1, size=n),
    }
)
toy_y = pd.Series(data=rng.integers(low=0, high=100, size=n))


def test_validation_method():
    f = lambda x: x  # noqa: E731
    f.__doc__ = "bla"
    vf = validation_method_X(f)
    assert getattr(vf, "_tags", {}).get("validation_X", False)
    assert vf.__doc__ == f.__doc__
    assert vf.__name__ == f.__name__

    def f2(x):
        """Bla"""
        return x

    vf = validation_method_X(f2)
    assert getattr(vf, "_tags", {}).get("validation_X", False)
    assert vf.__doc__ == f2.__doc__
    assert vf.__name__ == f2.__name__

    class F:
        @validation_method_X
        def f3(self, x):
            """Bla"""
            return x

    vf = F().f3
    assert getattr(vf, "_tags", {}).get("validation_X", False)
    assert F.f3.__doc__ == vf.__doc__
    assert F.f3.__name__ == vf.__name__
    assert vf.__doc__ == "Bla"
    assert vf.__name__ == "f3"


def test_validate_target():
    obj = ValidateTargetColumn()
    assert obj.validate_target(toy_y) is None


def test_validate_date_col():
    obj = ValidateDateColumn()
    obj.date_column = "date"
    assert obj.validate_date_col(toy_X) is None
    with pytest.raises(ValueError, match="date_col date not in data"):
        obj.validate_date_col(toy_X.drop(columns=["date"]))
    with pytest.raises(ValueError, match="date_col date has repeated values"):
        obj.validate_date_col(pd.concat([toy_X, toy_X], ignore_index=True, axis=0))


def test_channel_columns():
    obj = ValidateChannelColumns()
    obj.channel_columns = ["channel_1", "channel_2"]
    assert obj.validate_channel_columns(toy_X) is None

    with pytest.raises(ValueError, match="channel_columns must be a list or tuple"):
        obj.channel_columns = {}
        obj.validate_channel_columns(toy_X)
    with pytest.raises(ValueError, match="channel_columns must not be empty"):
        obj.channel_columns = []
        obj.validate_channel_columns(toy_X)
    with pytest.raises(
        ValueError,
        match=r"channel_columns \['out_of_columns'\] not in data",
    ):
        obj.channel_columns = ["out_of_columns"]
        obj.validate_channel_columns(toy_X)
    with pytest.raises(
        ValueError,
        match=r"channel_columns \['channel_1', 'channel_1'\] contains duplicates",
    ):
        obj.channel_columns = ["channel_1", "channel_1"]
        obj.validate_channel_columns(toy_X)
    with pytest.warns(
        UserWarning,
        match=r"channel_columns \['channel_1'\] contains negative values",
    ):
        new_toy_X = toy_X.copy()
        new_toy_X["channel_1"] -= 1e4
        obj.channel_columns = ["channel_1"]
        obj.validate_channel_columns(new_toy_X)


def test_control_columns():
    obj = ValidateControlColumns()
    obj.control_columns = None
    assert obj.validate_control_columns(toy_X) is None
    obj.control_columns = ["control_1", "control_2"]
    assert obj.validate_control_columns(toy_X) is None

    with pytest.raises(
        ValueError, match="control_columns must be None, a list or tuple"
    ):
        obj.control_columns = {}
        obj.validate_control_columns(toy_X)
    with pytest.raises(
        ValueError, match="If control_columns is not None, then it must not be empty"
    ):
        obj.control_columns = []
        obj.validate_control_columns(toy_X)
    with pytest.raises(
        ValueError,
        match=r"control_columns \['out_of_columns'\] not in data",
    ):
        obj.control_columns = ["out_of_columns"]
        obj.validate_control_columns(toy_X)
    with pytest.raises(
        ValueError,
        match=r"control_columns \['control_1', 'control_1'\] contains duplicates",
    ):
        obj.control_columns = ["control_1", "control_1"]
        obj.validate_control_columns(toy_X)


def test_y_len_0():
    obj = ValidateTargetColumn()
    with pytest.raises(ValueError, match="y must have at least one element"):
        obj.validate_target(pd.Series())
