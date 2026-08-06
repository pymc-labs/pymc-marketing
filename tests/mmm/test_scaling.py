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
    MaxAbsScaling,
    MeanAbsScaling,
    Scaling,
    VariableScaling,
    deserialize_variable_scaling,
    panel_channel_fixed_scaling_remaining_dims,
)
from pymc_marketing.serialization import serialization


class TestScalingRoundtrips:
    def test_data_derived_scaling_roundtrip(self):
        """Old DataDerivedScaling round-trips to itself via direct serialization."""
        original = DataDerivedScaling(method="mean", dims=("geo", "channel"))
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is DataDerivedScaling
        assert restored.method == "mean"
        assert restored.dims == ("geo", "channel")
        assert restored == original

    def test_max_abs_scaling_roundtrip(self):
        """New MaxAbsScaling round-trips to itself."""
        original = MaxAbsScaling(dims=("geo",))
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is MaxAbsScaling
        assert restored == original

    def test_mean_abs_scaling_roundtrip(self):
        """New MeanAbsScaling round-trips to itself."""
        original = MeanAbsScaling(dims=("geo", "channel"))
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is MeanAbsScaling
        assert restored == original

    def test_scaling_roundtrip_all_parameters(self):
        original = Scaling(
            target=MaxAbsScaling(dims="geo"),
            channel=MeanAbsScaling(dims=("geo", "channel")),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is Scaling
        assert type(restored.target) is MaxAbsScaling
        assert type(restored.channel) is MeanAbsScaling
        assert restored.target.dims == ("geo",)
        assert restored.channel.dims == ("geo", "channel")
        assert restored == original

    def test_scaling_roundtrip_mixed_types(self):
        original = Scaling(
            target=FixedScaling(dims=(), value=50_000.0),
            channel=MaxAbsScaling(dims=("geo",)),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is Scaling
        assert type(restored.target) is FixedScaling
        assert type(restored.channel) is MaxAbsScaling
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
        assert type(restored.target) is MaxAbsScaling
        assert type(restored.channel) is FixedScaling
        assert restored.target.dims == ("geo",)
        assert restored.channel.value == 1000.0

    def test_legacy_data_derived_max_roundtrip(self):
        """Old DataDerivedScaling(method='max') upgrades to MaxAbsScaling."""
        original = DataDerivedScaling(method="max", dims=())
        data = serialization.serialize(original)
        restored = deserialize_variable_scaling(data)

        assert type(restored) is MaxAbsScaling
        assert restored.dims == ()

    def test_legacy_data_derived_mean_roundtrip(self):
        """Old DataDerivedScaling(method='mean') upgrades to MeanAbsScaling."""
        original = DataDerivedScaling(method="mean", dims=("country",))
        data = serialization.serialize(original)
        restored = deserialize_variable_scaling(data)

        assert type(restored) is MeanAbsScaling
        assert restored.dims == ("country",)


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

    def test_fixed_scalar_nan_raises(self):
        with pytest.raises(ValueError, match="non-NaN"):
            FixedScaling(dims=(), value=float("nan"))

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

    def test_compute_scalar(self):
        vs = FixedScaling(dims=(), value=1000.0)
        data = xr.DataArray([1000.0, 2000.0], dims="date")
        arts = vs.compute(data)
        assert "scale" in arts
        assert float(arts["scale"]) == 1000.0

    def test_compute_dict(self):
        vs = FixedScaling(dims=(), value={"A": 10.0, "B": 20.0})
        data = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("date", "country"),
            coords={"country": ["A", "B"]},
        )
        arts = vs.compute(data)
        assert arts["scale"].sel(country="A").item() == 10.0
        assert arts["scale"].sel(country="B").item() == 20.0

    def test_compute_dataarray(self):
        user_da = xr.DataArray(
            [5.0, 10.0, 15.0],
            dims="channel",
            coords={"channel": ["tv", "radio", "search"]},
        )
        vs = FixedScaling(dims=(), value=user_da)
        data = xr.DataArray(
            np.ones((10, 3)),
            dims=("date", "channel"),
            coords={"channel": ["tv", "radio", "search"]},
        )
        arts = vs.compute(data)
        assert arts["scale"].sel(channel="tv").item() == 5.0
        assert arts["scale"].sel(channel="search").item() == 15.0


class TestDataDerivedScaling:
    def test_max_construction(self):
        vs = DataDerivedScaling(method="max", dims=())
        assert vs.method == "max"
        assert vs.dims == ()

    def test_mean_construction(self):
        vs = DataDerivedScaling(method="mean", dims=("geo",))
        assert vs.method == "mean"
        assert vs.dims == ("geo",)

    def test_compute(self):
        vs = DataDerivedScaling(method="max", dims=())
        data = xr.DataArray([3.0, 5.0, 1.0], dims="date")
        arts = vs.compute(data)
        assert float(arts["scale"]) == 5.0

    def test_transform(self):
        vs = DataDerivedScaling(method="max", dims=())
        data = xr.DataArray([2.0, 4.0, 8.0], dims="date")
        arts = vs.compute(data)
        result = vs.transform(data, arts)
        expected = data / 8.0
        xr.testing.assert_equal(result, expected)

    def test_inverse_transform(self):
        vs = DataDerivedScaling(method="max", dims=())
        data = xr.DataArray([2.0, 4.0, 8.0], dims="date")
        arts = vs.compute(data)
        scaled = vs.transform(data, arts)
        restored = vs.inverse_transform(scaled, arts)
        xr.testing.assert_allclose(restored, data)


class TestMaxAbsScaling:
    def test_construction(self):
        vs = MaxAbsScaling(dims=())
        assert vs.dims == ()

    def test_compute(self):
        vs = MaxAbsScaling(dims=())
        data = xr.DataArray([3.0, 5.0, 1.0], dims="date")
        arts = vs.compute(data)
        assert float(arts["scale"]) == 5.0

    def test_compute_multidimensional(self):
        vs = MaxAbsScaling(dims=("channel",))
        data = xr.DataArray(
            [[1.0, 3.0], [2.0, 4.0], [5.0, 1.0]],
            dims=("date", "channel"),
        )
        arts = vs.compute(data)
        # reduce over ("date", "channel") → scalar max
        assert float(arts["scale"]) == 5.0

    def test_compute_per_channel(self):
        vs = MaxAbsScaling(dims=())
        data = xr.DataArray(
            [[1.0, 3.0], [2.0, 4.0]],
            dims=("date", "channel"),
            coords={"channel": ["c1", "c2"]},
        )
        arts = vs.compute(data)
        # reduce over ("date",) → per-channel max
        expected = xr.Dataset(
            {
                "scale": xr.DataArray(
                    [2.0, 4.0], dims="channel", coords={"channel": ["c1", "c2"]}
                )
            }
        )
        xr.testing.assert_equal(arts, expected)

    def test_transform(self):
        vs = MaxAbsScaling(dims=())
        data = xr.DataArray([2.0, 4.0], dims="date")
        arts = vs.compute(data)
        result = vs.transform(data, arts)
        expected = data / 4.0
        xr.testing.assert_equal(result, expected)

    def test_inverse_transform(self):
        vs = MaxAbsScaling(dims=())
        data = xr.DataArray([2.0, 4.0], dims="date")
        arts = vs.compute(data)
        scaled = vs.transform(data, arts)
        restored = vs.inverse_transform(scaled, arts)
        xr.testing.assert_allclose(restored, data)


class TestMeanAbsScaling:
    def test_construction(self):
        vs = MeanAbsScaling(dims=())
        assert vs.dims == ()

    def test_compute(self):
        vs = MeanAbsScaling(dims=())
        data = xr.DataArray([2.0, 4.0, 6.0], dims="date")
        arts = vs.compute(data)
        assert float(arts["scale"]) == 4.0

    def test_transform(self):
        vs = MeanAbsScaling(dims=())
        data = xr.DataArray([2.0, 4.0], dims="date")
        arts = vs.compute(data)
        result = vs.transform(data, arts)
        expected = data / 3.0
        xr.testing.assert_equal(result, expected)


class TestVariableScalingIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            VariableScaling(dims=())

    def test_subclass_relationship(self):
        assert issubclass(DataDerivedScaling, VariableScaling)
        assert issubclass(FixedScaling, VariableScaling)
        assert issubclass(MaxAbsScaling, VariableScaling)
        assert issubclass(MeanAbsScaling, VariableScaling)


class TestVariableScalingDimsValidation:
    def test_date_dim_rejected_data_derived(self):
        with pytest.raises(ValueError, match="date"):
            DataDerivedScaling(method="max", dims=("date",))

    def test_date_dim_rejected_fixed(self):
        with pytest.raises(ValueError, match="date"):
            FixedScaling(dims=("date",), value=100.0)

    def test_date_dim_rejected_max_abs(self):
        with pytest.raises(ValueError, match="date"):
            MaxAbsScaling(dims=("date",))

    def test_duplicate_dims_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            DataDerivedScaling(method="max", dims=("geo", "geo"))


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.scaling.Scaling",
        "pymc_marketing.mmm.scaling.DataDerivedScaling",
        "pymc_marketing.mmm.scaling.FixedScaling",
        "pymc_marketing.mmm.scaling.MaxAbsScaling",
        "pymc_marketing.mmm.scaling.MeanAbsScaling",
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


def test_legacy_variable_scaling_type_key_deserializes():
    payload = {
        "__type__": "pymc_marketing.mmm.scaling.VariableScaling",
        "method": "max",
        "dims": ["geo"],
    }
    restored = Scaling.from_dict(
        {
            "target": payload,
            "channel": {
                "__type__": "pymc_marketing.mmm.scaling.VariableScaling",
                "method": "fixed",
                "dims": [],
                "value": 100.0,
            },
        }
    )
    assert isinstance(restored.target, MaxAbsScaling)
    assert restored.target.scaling_description() == "max-absolute"
    assert isinstance(restored.channel, FixedScaling)
    assert restored.channel.value == 100.0
