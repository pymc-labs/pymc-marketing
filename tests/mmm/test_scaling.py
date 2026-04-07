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
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm.scaling import (
    DataDerivedScaling,
    FixedScaling,
    Scaling,
    VariableScaling,
    panel_channel_fixed_scaling_remaining_dims,
)
from pymc_marketing.serialization import serialization


class TestScalingRoundtrips:
    def test_data_derived_scaling_roundtrip(self):
        original = DataDerivedScaling(method="mean", dims=("geo", "channel"))
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is DataDerivedScaling
        assert restored.method == "mean"
        assert restored.dims == ("geo", "channel")
        assert restored == original

    def test_scaling_roundtrip_all_parameters(self):
        original = Scaling(
            target=DataDerivedScaling(method="max", dims="geo"),
            channel=DataDerivedScaling(method="mean", dims=("geo", "channel")),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is Scaling
        assert type(restored.target) is DataDerivedScaling
        assert type(restored.channel) is DataDerivedScaling
        assert restored.target.method == "max"
        assert restored.target.dims == ("geo",)
        assert restored.channel.method == "mean"
        assert restored.channel.dims == ("geo", "channel")
        assert restored == original

    def test_scaling_roundtrip_mixed_types(self):
        original = Scaling(
            target=FixedScaling(dims=(), value=50_000.0),
            channel=DataDerivedScaling(method="max", dims=("geo",)),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is Scaling
        assert type(restored.target) is FixedScaling
        assert type(restored.channel) is DataDerivedScaling
        assert restored == original

    def test_legacy_format_deserialization(self):
        """Scaling.from_dict handles legacy dicts without __type__ keys."""
        legacy_data = {
            "__type__": "pymc_marketing.mmm.scaling.Scaling",
            "target": {"method": "max", "dims": ["geo"]},
            "channel": {"method": "fixed", "dims": [], "value": 1000.0},
        }
        restored = serialization.deserialize(legacy_data)

        assert type(restored) is Scaling
        assert type(restored.target) is DataDerivedScaling
        assert type(restored.channel) is FixedScaling
        assert restored.target.method == "max"
        assert restored.channel.value == 1000.0


class TestFixedScaling:
    def test_fixed_scalar_construction(self):
        vs = FixedScaling(dims=(), value=1000.0)
        assert vs.method == "fixed"
        assert vs.value == 1000.0
        assert vs.dims == ()

    def test_fixed_dict_construction(self):
        vs = FixedScaling(
            dims=("country",),
            value={"US": 50_000, "UK": 30_000},
        )
        assert vs.method == "fixed"
        assert vs.value == {"US": 50_000, "UK": 30_000}

    def test_fixed_zero_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            FixedScaling(dims=(), value=0.0)

    def test_fixed_negative_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            FixedScaling(dims=(), value=-5.0)

    def test_fixed_dict_negative_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            FixedScaling(dims=("geo",), value={"A": 100, "B": -1})

    def test_fixed_dict_nan_raises(self):
        with pytest.raises(ValueError, match="non-NaN"):
            FixedScaling(dims=("geo",), value={"US": float("nan")})

    def test_roundtrip_fixed_scalar(self):
        original = FixedScaling(dims=(), value=42.0)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert restored == original
        assert restored.value == 42.0

    def test_roundtrip_fixed_dict(self):
        original = FixedScaling(
            dims=("geo",),
            value={"US": 100, "UK": 200},
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert restored == original
        assert restored.value == {"US": 100, "UK": 200}

    def test_roundtrip_fixed_dataarray(self):
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims="country",
            coords={"country": ["A", "B", "C"]},
        )
        original = FixedScaling(dims=(), value=da)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert isinstance(restored, FixedScaling)
        assert isinstance(restored.value, xr.DataArray)
        xr.testing.assert_equal(restored.value, da)

    def test_fixed_dataarray_nan_raises(self):
        da = xr.DataArray([1.0, np.nan], dims="x", coords={"x": [0, 1]})
        with pytest.raises(ValueError, match="NaN"):
            FixedScaling(dims=(), value=da)

    def test_fixed_bool_raises(self):
        with pytest.raises(ValidationError):
            FixedScaling(dims=(), value=True)

    def test_fixed_scaling_invalid_value_type_raises(self):
        with pytest.raises(ValidationError):
            FixedScaling(dims=(), value="nope")  # type: ignore[arg-type]

    def test_from_long_dataframe(self):
        df = pd.DataFrame(
            {
                "country": ["A", "A", "B", "B"],
                "channel": ["c1", "c2", "c1", "c2"],
                "scale": [10.0, 20.0, 30.0, 40.0],
            }
        )
        fs = FixedScaling.from_long_dataframe(
            dims=(),
            df=df,
            value_col="scale",
            dim_cols=["country", "channel"],
        )
        assert isinstance(fs.value, xr.DataArray)
        assert fs.value.sizes == {"country": 2, "channel": 2}
        assert float(fs.value.sel(country="A", channel="c1")) == 10.0
        round_data = serialization.serialize(fs)
        restored = serialization.deserialize(round_data)
        xr.testing.assert_equal(restored.value, fs.value)

    def test_from_long_dataframe_duplicate_rows_raises(self):
        df = pd.DataFrame(
            {
                "country": ["A", "A", "A"],
                "channel": ["c1", "c1", "c2"],
                "scale": [10.0, 99.0, 30.0],
            }
        )
        with pytest.raises(ValueError, match="Duplicate coordinate rows"):
            FixedScaling.from_long_dataframe(
                dims=(),
                df=df,
                value_col="scale",
                dim_cols=["country", "channel"],
            )

    def test_roundtrip_scaling_with_fixed(self):
        original = Scaling(
            target=FixedScaling(dims=(), value=50_000.0),
            channel=FixedScaling(dims=(), value=10_000.0),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert restored == original


class TestDataDerivedScaling:
    def test_max_construction(self):
        vs = DataDerivedScaling(method="max", dims=())
        assert vs.method == "max"
        assert vs.dims == ()

    def test_mean_construction(self):
        vs = DataDerivedScaling(method="mean", dims=("geo",))
        assert vs.method == "mean"
        assert vs.dims == ("geo",)


class TestVariableScalingIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            VariableScaling(dims=())

    def test_subclass_relationship(self):
        assert issubclass(DataDerivedScaling, VariableScaling)
        assert issubclass(FixedScaling, VariableScaling)


class TestVariableScalingDimsValidation:
    def test_date_dim_rejected_data_derived(self):
        with pytest.raises(ValueError, match="date"):
            DataDerivedScaling(method="max", dims=("date",))

    def test_date_dim_rejected_fixed(self):
        with pytest.raises(ValueError, match="date"):
            FixedScaling(dims=("date",), value=100.0)

    def test_duplicate_dims_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            DataDerivedScaling(method="max", dims=("geo", "geo"))


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.scaling.Scaling",
        "pymc_marketing.mmm.scaling.DataDerivedScaling",
        "pymc_marketing.mmm.scaling.FixedScaling",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_scaling_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"


def test_panel_channel_fixed_scaling_remaining_dims():
    assert panel_channel_fixed_scaling_remaining_dims(("country",), ("country",)) == (
        "channel",
    )
    assert panel_channel_fixed_scaling_remaining_dims(("country",), ()) == (
        "country",
        "channel",
    )


def test_abstract_variable_scaling_not_registered():
    assert "pymc_marketing.mmm.scaling.VariableScaling" not in serialization._registry
