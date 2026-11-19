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

from pymc_marketing.conversion import to_funnel_dataset

STAGES = ["visit", "signup", "trial", "paid"]


def stage_counts() -> pd.DataFrame:
    """Stage counts of a closed funnel for two variants, in shuffled row order."""
    df = pd.DataFrame(
        {
            "stage": STAGES * 2,
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "count": [1000, 400, 200, 50, 1000, 450, 260, 80],
        }
    )
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


def funnel_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "entered": (("step", "variant"), [[1000, 1000], [400, 450], [200, 260]]),
            "converted": (("step", "variant"), [[400, 450], [200, 260], [50, 80]]),
        },
        coords={"step": STAGES[1:], "variant": ["control", "treatment"]},
    )


def test_stage_counts_to_entered_and_converted() -> None:
    ds = to_funnel_dataset(stage_counts(), stage_order=STAGES)

    xr.testing.assert_identical(ds, funnel_dataset().astype("int64"))
    assert ds["entered"].dims == ("step", "variant")
    assert ds["converted"].dtype == np.int64


def test_order_from_ordered_categorical() -> None:
    df = stage_counts()
    df["stage"] = pd.Categorical(df["stage"], categories=STAGES, ordered=True)

    xr.testing.assert_identical(
        to_funnel_dataset(df), to_funnel_dataset(stage_counts(), stage_order=STAGES)
    )


def test_no_group_columns() -> None:
    df = stage_counts().query("variant == 'control'").drop(columns="variant")

    ds = to_funnel_dataset(df, stage_order=STAGES)

    assert ds["converted"].dims == ("step",)
    np.testing.assert_array_equal(ds["entered"], [1000, 400, 200])
    np.testing.assert_array_equal(ds["converted"], [400, 200, 50])


def test_two_group_columns_keep_column_order() -> None:
    df = pd.concat(
        [stage_counts().assign(segment=segment) for segment in ["new", "returning"]]
    )[["segment", "stage", "variant", "count"]]

    ds = to_funnel_dataset(df, stage_order=STAGES)

    assert ds["converted"].dims == ("step", "segment", "variant")
    assert ds.sizes == {"step": 3, "segment": 2, "variant": 2}


def test_integer_group_labels_are_kept() -> None:
    df = stage_counts()
    df["variant"] = df["variant"].map({"control": 10, "treatment": 20})

    ds = to_funnel_dataset(df, stage_order=STAGES)

    assert list(ds["variant"].values) == [10, 20]
    assert np.issubdtype(ds["variant"].dtype, np.integer)


def test_dataset_is_validated_and_step_goes_first() -> None:
    ds = funnel_dataset().astype("float64").transpose("variant", "step")

    result = to_funnel_dataset(ds)

    xr.testing.assert_identical(result, funnel_dataset().astype("int64"))


def test_dataset_keeps_other_variables() -> None:
    ds = funnel_dataset().assign(price=("variant", [9.0, 12.0]))

    result = to_funnel_dataset(ds)

    np.testing.assert_array_equal(result["price"], [9.0, 12.0])


def test_zero_entered_cell_is_allowed() -> None:
    df = stage_counts()
    df.loc[
        (df["variant"] == "control") & df["stage"].isin(["trial", "paid"]), "count"
    ] = 0

    ds = to_funnel_dataset(df, stage_order=STAGES)

    assert ds["entered"].sel(step="paid", variant="control").item() == 0
    assert ds["converted"].sel(step="paid", variant="control").item() == 0


def test_categorical_group_column_gives_a_dataset_that_can_be_saved(tmp_path) -> None:
    df = stage_counts()
    df["stage"] = pd.Categorical(df["stage"], categories=STAGES, ordered=True)
    df["variant"] = df["variant"].astype("category")

    ds = to_funnel_dataset(df)

    xr.testing.assert_identical(ds, funnel_dataset().astype("int64"))
    assert all(isinstance(ds[dim].dtype, np.dtype) for dim in ds.dims)
    ds.to_netcdf(tmp_path / "funnel.nc")


def _unlabelled_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["variant"] = df["variant"].astype(object)
    df.loc[(df["stage"] == "paid") & (df["variant"] == "treatment"), "variant"] = None
    return df


def _drop_row(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df["stage"] == "trial") & (df["variant"] == "control"))]


def _set_count(df: pd.DataFrame, stage: str, value: float) -> pd.DataFrame:
    df = df.copy()
    df["count"] = df["count"].astype("float64")
    df.loc[(df["stage"] == stage) & (df["variant"] == "control"), "count"] = value
    return df


@pytest.mark.parametrize(
    "data, kwargs, match",
    [
        pytest.param(
            stage_counts().drop(columns="count"),
            {"stage_order": STAGES},
            "missing \\['count'\\]",
            id="missing column",
        ),
        pytest.param(
            pd.concat([stage_counts(), stage_counts().head(1)]),
            {"stage_order": STAGES},
            "more than one row",
            id="duplicated rows",
        ),
        pytest.param(
            _unlabelled_row(stage_counts()),
            {"stage_order": STAGES},
            "\\['variant'\\] have rows without a label",
            id="missing group label",
        ),
        pytest.param(stage_counts(), {}, "order of the stages", id="no stage order"),
        pytest.param(
            stage_counts(),
            {"stage_order": ["visit", "signup", "trial", "purchase"]},
            "not in the data: \\['purchase'\\].*not in stage_order: \\['paid'\\]",
            id="order and labels differ",
        ),
        pytest.param(
            stage_counts(),
            {"stage_order": ["visit", "signup", "trial", "paid", "paid"]},
            "stage_order repeats",
            id="repeated stage in order",
        ),
        pytest.param(
            _drop_row(stage_counts()),
            {"stage_order": STAGES},
            "no count for stage='trial', variant='control'",
            id="missing cell",
        ),
        pytest.param(
            stage_counts().query("stage == 'visit'"),
            {"stage_order": ["visit"]},
            "at least two stages",
            id="one stage",
        ),
        pytest.param(
            stage_counts(),
            {"stage_order": STAGES[::-1]},
            "increases from stage 'paid' to stage 'trial' for variant='control'.*Check stage_order",
            id="counts increase",
        ),
        pytest.param(
            _set_count(stage_counts(), "paid", -1.0),
            {"stage_order": STAGES},
            "negative",
            id="negative count",
        ),
        pytest.param(
            _set_count(stage_counts(), "paid", 12.5),
            {"stage_order": STAGES},
            "whole numbers",
            id="fractional count",
        ),
        pytest.param(
            _set_count(stage_counts(), "paid", np.nan),
            {"stage_order": STAGES},
            "no count for stage='paid', variant='control'",
            id="nan count",
        ),
        pytest.param(
            funnel_dataset().drop_vars("entered"),
            {},
            "missing \\['entered'\\]",
            id="dataset missing variable",
        ),
        pytest.param(
            funnel_dataset().assign(entered=("step", [1000, 400, 200])),
            {},
            "same dimensions",
            id="dataset dims differ",
        ),
        pytest.param(
            funnel_dataset().rename(step="stage"),
            {},
            "'step' dimension",
            id="dataset without step",
        ),
        pytest.param(
            funnel_dataset().drop_vars("variant"),
            {},
            "'variant' has no coordinate",
            id="dataset dim without coordinate",
        ),
        pytest.param(
            funnel_dataset().assign(
                converted=(("step", "variant"), [[400, 450], [200, 500], [50, 80]])
            ),
            {},
            "more conversions than entrants in 1 cell.*step='trial', variant='treatment'",
            id="converted exceeds entered",
        ),
        pytest.param(
            funnel_dataset().assign(
                entered=(("step", "variant"), [[1000, 1000], [400, 450], [150, 260]])
            ),
            {},
            "not describe a closed funnel: in 1 cell.*step='paid', variant='control'",
            id="entered differs from converted at the step before",
        ),
        pytest.param(
            funnel_dataset(),
            {"stage_order": STAGES},
            "stage_order applies to a DataFrame",
            id="stage_order with dataset",
        ),
    ],
)
def test_value_errors(data, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        to_funnel_dataset(data, **kwargs)


def test_unsupported_type() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame or an xarray Dataset"):
        to_funnel_dataset(np.array([100, 50, 10]))  # type: ignore[arg-type]
