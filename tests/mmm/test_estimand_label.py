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
"""Tests for the label an incrementality result carries.

No model is involved: a label is a pydantic model, and what binds it to an
array is a digest of that array.
"""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm.estimand_label import LABEL_ATTR, EstimandLabel

REQUEST = {
    "scope": "per_channel",
    "frequency": "monthly",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-03-31T00:00:00",
    "counterfactual_spend_factor": 0.0,
    "include_carryover": True,
}
RESPONSE = {**REQUEST, "central_tendency": "median", "accounting_verified": True}
TOTAL = {**RESPONSE, "quantity": "incremental_contribution", "paths": "total"}
DIRECT = {
    **RESPONSE,
    "quantity": "incremental_contribution",
    "paths": "direct",
    "held": ("funnel_effect_contribution",),
    "held_at": "factual",
}
MEDIATOR = {
    **REQUEST,
    "quantity": "incremental_mediator",
    "mediator": "lf_spend",
    "variable": "funnel_spend",
    "units": "USD",
    "is_spend": True,
}


def result(seed: int = 0, *, complete: bool = False) -> xr.DataArray:
    """A stand-in for a result, with the dimensions and label kinds of a real one."""
    coords = {
        "chain": [0, 1],
        "draw": [3, 7, 11],
        "date": pd.date_range("2024-01-31", periods=3, freq="ME"),
        "channel": ["tv", "social"],
        "geo": ["north", "south"],
    }
    shape = tuple(len(values) for values in coords.values())
    array = xr.DataArray(
        np.random.default_rng(seed).normal(size=shape),
        dims=tuple(coords),
        coords=coords,
    )
    if complete:
        array = array.assign_coords(complete=("date", [True, True, False]))
    return array


class TestCoherence:
    """A label that contradicts itself cannot be built."""

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param(TOTAL, id="a total"),
            pytest.param(DIRECT, id="a direct increment"),
            pytest.param(
                {**DIRECT, "paths": "mediated", "held": ()},
                id="a mediated increment with nothing held",
            ),
            pytest.param(
                {**TOTAL, "accounting_verified": False},
                id="a total returned unverified",
            ),
            pytest.param(MEDIATOR, id="a mediator movement"),
            pytest.param(
                {**REQUEST, "quantity": "spend_change"}, id="a change in spend"
            ),
            pytest.param(
                {
                    **REQUEST,
                    "quantity": "induced_spend",
                    "induced_spend": ("lf_spend",),
                    "induced_spend_units": "USD",
                },
                id="induced spend",
            ),
            pytest.param(
                {
                    **TOTAL,
                    "quantity": "contribution_over_spend",
                    "induced_spend": ("lf_spend",),
                    "induced_spend_units": "USD",
                },
                id="a ratio over all incremental spend",
            ),
            pytest.param(
                {
                    **TOTAL,
                    "quantity": "marginal_contribution_over_spend",
                    "counterfactual_spend_factor": 1.01,
                },
                id="a marginal ratio",
            ),
        ],
    )
    def test_what_the_module_issues_is_coherent(self, fields):
        """Every kind of label a public method attaches validates."""
        assert EstimandLabel(**fields).quantity == fields["quantity"]

    @pytest.mark.parametrize(
        "fields, complaint",
        [
            pytest.param(
                {**DIRECT, "accounting_verified": False},
                "never returned unverified",
                id="a direct increment that was not verified",
            ),
            pytest.param(
                {**TOTAL, "held": ("funnel_effect_contribution",)},
                "only a direct or mediated path holds",
                id="a total that holds something",
            ),
            pytest.param(
                {**DIRECT, "held_at": None},
                "held_at",
                id="a direct increment that does not say where mediators are held",
            ),
            pytest.param(
                {**DIRECT, "held": ("b_term", "a_term")},
                "sorted",
                id="held out of order",
            ),
            pytest.param(
                {**TOTAL, "central_tendency": None},
                "a response result has",
                id="a response with no central tendency",
            ),
            pytest.param(
                {**MEDIATOR, "paths": "total"},
                "only a response result has",
                id="a mediator movement with paths",
            ),
            pytest.param(
                {**MEDIATOR, "variable": None},
                "names its mediator",
                id="a mediator movement with no variable",
            ),
            pytest.param(
                {**MEDIATOR, "units": None},
                "a spend mediator has units",
                id="a spend mediator with no units",
            ),
            pytest.param(
                {**TOTAL, "mediator": "lf_spend"},
                "belong to a mediator result",
                id="a response that names a mediator",
            ),
            pytest.param(
                {**TOTAL, "induced_spend": ("lf_spend",), "induced_spend_units": "USD"},
                "denominator of a total ratio",
                id="induced spend on an increment",
            ),
            pytest.param(
                {
                    **DIRECT,
                    "quantity": "contribution_over_spend",
                    "induced_spend": ("lf_spend",),
                    "induced_spend_units": "USD",
                },
                "denominator of a total ratio",
                id="induced spend under a direct ratio",
            ),
            pytest.param(
                {**TOTAL, "quantity": "contribution_over_spend", "scope": "joint"},
                "a ratio is per channel",
                id="a joint ratio",
            ),
            pytest.param(
                {
                    **DIRECT,
                    "quantity": "spend_over_contribution",
                    "paths": "mediated",
                },
                "not offered",
                id="spend over a mediated share",
            ),
            pytest.param(
                {**TOTAL, "quantity": "marginal_contribution_over_spend"},
                "a marginal ratio raises spend",
                id="a marginal ratio that removes spend",
            ),
            pytest.param(
                {
                    **TOTAL,
                    "quantity": "contribution_over_spend",
                    "counterfactual_spend_factor": 0.5,
                },
                "an average ratio removes spend",
                id="an average ratio at another factor",
            ),
            pytest.param(
                {**TOTAL, "start_date": "the first of January"},
                "ISO 8601",
                id="a period that is not a date",
            ),
            pytest.param(
                {**TOTAL, "start_date": "2024-06-01T00:00:00"},
                "starts before it ends",
                id="a period that ends before it starts",
            ),
            pytest.param(
                {**TOTAL, "schema_version": 7}, "schema version", id="a version"
            ),
        ],
    )
    def test_a_contradiction_is_named(self, fields, complaint):
        """Each rule has its own sentence in the error."""
        with pytest.raises(ValidationError, match=complaint):
            EstimandLabel(**fields)

    def test_every_contradiction_is_reported_at_once(self):
        """A label with two faults says both, so one fix does not hide the other."""
        with pytest.raises(ValidationError) as error:
            EstimandLabel(**{**DIRECT, "accounting_verified": False, "held_at": None})

        assert "never returned unverified" in str(error.value)
        assert "held_at" in str(error.value)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("counterfactual_spend_factor", "0.0"),
            ("counterfactual_spend_factor", float("inf")),
            ("counterfactual_spend_factor", -1.0),
            ("include_carryover", 1),
            ("accounting_verified", "true"),
            ("start_date", 20240101),
            ("held", ("",)),
            ("definition", "anything"),
        ],
    )
    def test_nothing_is_coerced(self, field, value):
        """A label is written by code and read from a file, never typed loosely."""
        with pytest.raises(ValidationError):
            EstimandLabel(**{**DIRECT, field: value})


class TestDefinition:
    """The contrast in symbols, with the sign the result carries."""

    @pytest.mark.parametrize(
        "fields, definition",
        [
            (TOTAL, "Y(x) - Y(x')"),
            ({**TOTAL, "counterfactual_spend_factor": 1.5}, "Y(x') - Y(x)"),
            (DIRECT, "Y(x) - Y(x', M(x))"),
            ({**DIRECT, "counterfactual_spend_factor": 1.01}, "Y(x', M(x)) - Y(x)"),
            ({**DIRECT, "paths": "mediated"}, "Y(x', M(x)) - Y(x')"),
            (
                {**DIRECT, "paths": "mediated", "counterfactual_spend_factor": 1.01},
                "Y(x') - Y(x', M(x))",
            ),
            (MEDIATOR, "M(x) - M(x')"),
            ({**MEDIATOR, "counterfactual_spend_factor": 2.0}, "M(x') - M(x)"),
            ({**REQUEST, "quantity": "spend_change"}, "S(x) - S(x')"),
            (
                {**TOTAL, "quantity": "contribution_over_spend"},
                "[Y(x) - Y(x')] / [S(x) - S(x')]",
            ),
            (
                {**TOTAL, "quantity": "spend_over_contribution"},
                "[S(x) - S(x')] / [Y(x) - Y(x')]",
            ),
            (
                {
                    **TOTAL,
                    "quantity": "contribution_over_spend",
                    "induced_spend": ("lf_spend",),
                    "induced_spend_units": "USD",
                },
                "[Y(x) - Y(x')] / [S(x) - S(x') + M(x) - M(x')]",
            ),
            (
                {
                    **DIRECT,
                    "quantity": "marginal_contribution_over_spend",
                    "counterfactual_spend_factor": 1.01,
                },
                "[Y(x', M(x)) - Y(x)] / [S(x') - S(x)]",
            ),
        ],
    )
    def test_the_definition_follows_the_fields(self, fields, definition):
        """Above a factor of one the treated arm is the counterfactual."""
        assert EstimandLabel(**fields).definition == definition


class TestBoundToTheArray:
    """A label reads back from the array it was issued for, and from no other."""

    def test_attach_leaves_one_attribute_and_the_values_alone(self):
        """Whatever the array carried before, it now says one thing."""
        array = result().assign_attrs(units="USD", estimand="something else")

        labelled = EstimandLabel(**TOTAL).attach(array)

        assert list(labelled.attrs) == [LABEL_ATTR]
        assert array.attrs == {"units": "USD", "estimand": "something else"}
        np.testing.assert_array_equal(labelled.values, array.values)
        stored = json.loads(labelled.attrs[LABEL_ATTR])
        assert stored["definition"] == "Y(x) - Y(x')"
        assert stored["paths"] == "total"

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(lambda a: a, id="the array itself"),
            pytest.param(lambda a: a.copy(deep=True), id="a copy"),
            pytest.param(
                lambda a: a.transpose("geo", "channel", "date", "draw", "chain"),
                id="a transpose",
            ),
            pytest.param(lambda a: a.rename("anything"), id="a new name"),
            pytest.param(
                lambda a: a.assign_coords(region=("geo", ["a", "b"])),
                id="a new coordinate off the dimensions",
            ),
        ],
    )
    def test_the_same_result_reads_back(self, change):
        """None of these changes a value, a label or the order along a dimension."""
        label = EstimandLabel(**DIRECT)

        assert EstimandLabel.read(change(label.attach(result(complete=True)))) == label

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(lambda a: 2 * a, id="arithmetic"),
            pytest.param(
                lambda a: a.assign_coords(channel=["social", "tv"]),
                id="channel labels swapped",
            ),
            pytest.param(
                lambda a: a.assign_coords(draw=[0, 1, 2]), id="draws renumbered"
            ),
            pytest.param(
                lambda a: a.assign_coords(
                    date=a["date"].to_index() + pd.offsets.MonthEnd(1)
                ),
                id="dates shifted",
            ),
            pytest.param(
                lambda a: a.assign_coords(complete=("date", [True, True, True])),
                id="the complete mark flipped",
            ),
            pytest.param(
                lambda a: a.drop_vars("complete"), id="the complete mark dropped"
            ),
            pytest.param(lambda a: a.sel(channel=["tv"]), id="a selection"),
            pytest.param(
                lambda a: a.sel(channel="tv"), id="a selection that drops a dim"
            ),
            pytest.param(lambda a: a.isel(date=slice(0, 2)), id="a cut along date"),
            pytest.param(lambda a: a.sortby("channel"), id="a sort"),
            pytest.param(
                lambda a: a.mean(("chain", "draw"), keep_attrs=True), id="a reduction"
            ),
            pytest.param(
                lambda a: xr.concat([a, a], "model", combine_attrs="override"),
                id="a concatenation",
            ),
        ],
    )
    def test_anything_derived_is_refused(self, change):
        """A label is read before a result is cut, reordered or computed with."""
        labelled = EstimandLabel(**DIRECT).attach(result(complete=True))
        derived = change(labelled).assign_attrs(labelled.attrs)

        with pytest.raises(ValueError, match="do not belong together"):
            EstimandLabel.read(derived)

    def test_a_text_label_is_not_the_number_it_spells(self):
        """Geo ``1`` and geo ``'1'`` are two labels."""
        labelled = EstimandLabel(**TOTAL).attach(result().assign_coords(geo=[1, 2]))

        with pytest.raises(ValueError, match="do not belong together"):
            EstimandLabel.read(labelled.assign_coords(geo=["1", "2"]))

    @pytest.mark.parametrize(
        "edit",
        [
            {"paths": "total", "held": [], "held_at": None},
            {"held": ["another_effect_contribution"]},
            {"counterfactual_spend_factor": 1.01},
            {"frequency": "weekly"},
            {"end_date": "2024-06-30T00:00:00"},
            {"central_tendency": "mean"},
            {"scope": "joint"},
            {"include_carryover": False},
        ],
    )
    def test_an_edited_label_is_refused(self, edit):
        """The digest covers the stored fields, so a label cannot be relabelled."""
        labelled = EstimandLabel(**DIRECT).attach(result())
        stored = json.loads(labelled.attrs[LABEL_ATTR])
        labelled.attrs[LABEL_ATTR] = json.dumps({**stored, **edit})

        with pytest.raises(ValueError, match="do not belong together"):
            EstimandLabel.read(labelled)

    def test_a_reworded_definition_still_reads(self):
        """The stored text is for a reader of the raw attribute, and is regenerated."""
        label = EstimandLabel(**DIRECT)
        labelled = label.attach(result())
        stored = json.loads(labelled.attrs[LABEL_ATTR])
        labelled.attrs[LABEL_ATTR] = json.dumps({**stored, "definition": "reworded"})

        assert EstimandLabel.read(labelled) == label
        assert EstimandLabel.read(labelled).definition == "Y(x) - Y(x', M(x))"

    @pytest.mark.parametrize(
        "attrs, complaint",
        [
            pytest.param({}, "No 'incrementality_label'", id="no label"),
            pytest.param({LABEL_ATTR: "{not json"}, "not JSON", id="not JSON"),
            pytest.param(
                {LABEL_ATTR: "[1, 2]"}, "not a JSON object", id="not an object"
            ),
            pytest.param(
                {LABEL_ATTR: json.dumps({"paths": "total"})},
                "schema version",
                id="no version",
            ),
        ],
    )
    def test_what_is_not_a_label_is_refused_as_a_value_error(self, attrs, complaint):
        """Every refusal of ``read`` is a ``ValueError``."""
        with pytest.raises(ValueError, match=complaint):
            EstimandLabel.read(result().assign_attrs(attrs))

    def test_a_dataset_is_told_to_read_a_variable(self):
        """A ``Dataset`` carries a label on each variable and none of its own."""
        dataset = EstimandLabel(**TOTAL).attach(result()).to_dataset(name="total")

        with pytest.raises(ValueError, match="variable of the Dataset"):
            EstimandLabel.read(dataset)


def _engines() -> list[str]:
    return [name for name in ("h5netcdf", "scipy") if importlib.util.find_spec(name)]


class TestThroughAFile:
    """The digest is the same before and after a file."""

    @staticmethod
    def labelled() -> xr.DataArray:
        """A ratio with a missing cell, as a channel with no spend leaves one."""
        array = result(complete=True)
        array.values[0, 1, 2, 0, 1] = np.nan
        return EstimandLabel(**{**TOTAL, "quantity": "contribution_over_spend"}).attach(
            array
        )

    @pytest.mark.parametrize("engine", _engines())
    def test_netcdf(self, tmp_path, engine):
        """Dates, text and integer labels, ``complete`` and a NaN all come back."""
        labelled = self.labelled()
        path = tmp_path / "result.nc"

        labelled.to_netcdf(path, engine=engine)
        with xr.open_dataarray(path, engine=engine) as stored:
            assert EstimandLabel.read(stored.load()) == EstimandLabel.read(labelled)

    def test_zarr(self, tmp_path):
        """A label on a variable of a ``Dataset`` survives a zarr store."""
        pytest.importorskip("zarr")
        labelled = self.labelled()
        path = tmp_path / "result.zarr"

        try:
            labelled.to_dataset(name="roas").to_zarr(path, consolidated=False)
        except Exception as error:  # an xarray too old for the installed zarr
            pytest.skip(f"this xarray cannot write the installed zarr: {error!r}")
        with xr.open_zarr(path, consolidated=False) as stored:
            assert EstimandLabel.read(stored["roas"].load()) == EstimandLabel.read(
                labelled
            )


class _NextSchema(EstimandLabel):
    """What a later release would look like: one field renamed, another digest rule."""

    SCHEMA_VERSION = 2
    MIGRATIONS = {1: lambda fields: {**fields, "schema_version": 2}}
    DIGESTS = {**EstimandLabel.DIGESTS, 2: lambda array, fields: "rule of version 2"}

    schema_version: int = 2


class TestVersions:
    """A file written by one release is read by the next, and not the reverse."""

    def test_every_older_version_has_a_migration_and_a_digest_rule(self):
        """A schema change that ships without them would orphan stored results."""
        versions = range(1, EstimandLabel.SCHEMA_VERSION + 1)

        assert set(EstimandLabel.DIGESTS) == set(versions)
        assert set(EstimandLabel.MIGRATIONS) == set(versions[:-1])

    def test_an_older_label_is_verified_by_its_own_rule_then_migrated(self):
        """The digest is checked on the fields as stored, before anything is upgraded."""
        old = EstimandLabel(**DIRECT).attach(result())

        upgraded = _NextSchema.read(old)

        assert upgraded.schema_version == 2
        assert upgraded.paths == "direct"

    def test_a_newer_label_asks_for_an_upgrade(self):
        """Code that does not know a schema cannot say what its fields mean."""
        new = _NextSchema(**{**DIRECT, "schema_version": 2}).attach(result())

        with pytest.raises(ValueError, match="upgrade pymc-marketing"):
            EstimandLabel.read(new)

    def test_a_missing_migration_is_named(self):
        """A gap in the chain is an error of the release, said as one."""

        class Gap(_NextSchema):
            MIGRATIONS = {}

        with pytest.raises(ValueError, match="No migration from label schema 1"):
            Gap.read(EstimandLabel(**DIRECT).attach(result()))

    def test_a_truncated_label_never_reaches_a_migration(self):
        """A payload that lost a field fails its digest first."""
        old = EstimandLabel(**DIRECT).attach(result())
        stored = json.loads(old.attrs[LABEL_ATTR])
        del stored["held_at"]
        old.attrs[LABEL_ATTR] = json.dumps(stored)

        with pytest.raises(ValueError, match="do not belong together"):
            _NextSchema.read(old)
