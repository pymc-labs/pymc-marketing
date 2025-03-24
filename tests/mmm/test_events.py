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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from pymc_marketing.mmm.events import (
    EventEffect,
    GaussianBasis,
    basis_from_dict,
    days_from_reference,
)
from pymc_marketing.plot import plot_curve
from pymc_marketing.prior import Prior


def test_gaussian_basis_plot() -> None:
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=[4, 7, 10], sigma=1, dims="event"),
        },
    )
    coords = {"event": ["NYE", "Grand Opening Game Show", "Super Bowl"]}
    prior = gaussian.sample_prior(coords=coords)
    curve = gaussian.sample_curve(prior, days=21)

    fig, _ = gaussian.plot_curve(curve, same_axes=True)
    plt.close()


def test_event_basis_in_model() -> None:
    df_events = pd.DataFrame(
        {
            "event": ["first", "second"],
            "start_date": pd.to_datetime(["2023-01-01", "2023-01-20"]),
            "end_date": pd.to_datetime(["2023-01-02", "2023-01-25"]),
        }
    )

    def difference_in_days(model_dates, event_dates):
        if hasattr(model_dates, "to_numpy"):
            model_dates = model_dates.to_numpy()
        if hasattr(event_dates, "to_numpy"):
            event_dates = event_dates.to_numpy()

        one_day = np.timedelta64(1, "D")
        return (model_dates[:, None] - event_dates) / one_day

    def create_basis_matrix(df_events: pd.DataFrame, model_dates: np.ndarray):
        start_dates = df_events["start_date"]
        end_dates = df_events["end_date"]

        s_ref = difference_in_days(model_dates, start_dates)
        e_ref = difference_in_days(model_dates, end_dates)

        return np.where(
            (s_ref >= 0) & (e_ref <= 0),
            0,
            np.where(np.abs(s_ref) < np.abs(e_ref), s_ref, e_ref),
        )

    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=7, sigma=1, dims="event"),
        }
    )
    effect_size = Prior("Normal", mu=1, sigma=1, dims="event")
    effect = EventEffect(basis=gaussian, effect_size=effect_size, dims=("event",))

    dates = pd.date_range("2022-12-01", periods=3 * 31, freq="D")

    X = create_basis_matrix(df_events, model_dates=dates)

    coords = {"date": dates, "event": df_events["event"].to_numpy()}
    with pm.Model(coords=coords):
        pm.Deterministic("effect", effect.apply(X), dims=("date", "event"))

        idata = pm.sample_prior_predictive()

    idata.prior.effect.pipe(
        plot_curve,
        {"date"},
        subplot_kwargs={"ncols": 1},
    )
    plt.close()


def test_gaussian_basis_serialization():
    # Test serialization/deserialization of GaussianBasis
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=7, sigma=1),
        },
    )

    # Test to_dict and from_dict
    gaussian_dict = gaussian.to_dict()
    gaussian_restored = basis_from_dict(gaussian_dict)

    assert gaussian_restored.lookup_name == gaussian.lookup_name
    assert gaussian_restored.prefix == gaussian.prefix


def test_event_effect_serialization():
    # Test serialization/deserialization of EventEffect
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=7, sigma=1),
        },
    )
    effect_size = Prior("Normal", mu=1, sigma=1, dims="event")
    effect = EventEffect(basis=gaussian, effect_size=effect_size, dims=("event",))

    # Test to_dict and from_dict
    effect_dict = effect.to_dict()
    effect_restored = EventEffect.from_dict(effect_dict["data"])

    assert effect_restored.dims == effect.dims
    assert effect_restored.basis.lookup_name == effect.basis.lookup_name
    assert effect_restored.effect_size.to_dict() == effect.effect_size.to_dict()


def test_gaussian_basis_curve_sampling():
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=7, sigma=1),
        },
    )

    # Test curve sampling with different days
    parameters = xr.Dataset(
        {"sigma": (["chain", "draw"], np.array([[7.0]]))},
        coords={
            "chain": [0],
            "draw": [0],
        },
    )

    curve_10 = gaussian.sample_curve(parameters, days=10)

    assert isinstance(curve_10, xr.DataArray)
    assert len(curve_10.x) == 100  # Check default number of points
    assert curve_10.x.min() == -10  # Check range
    assert curve_10.x.max() == 10


def test_gaussian_basis_function():
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=7, sigma=1),
        },
    )

    # Test the Gaussian function directly
    x = np.array([0.0, 1.0, -1.0])
    sigma = np.array([1.0])

    result = gaussian.function(x, sigma).eval()
    expected = np.exp(-0.5 * (x / sigma) ** 2)

    np.testing.assert_array_almost_equal(result, expected)


def test_gaussian_basis_multiple_events():
    # Test GaussianBasis with multiple events
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=[5, 8], sigma=1, dims="event"),
        },
    )
    coords = {"event": ["Event1", "Event2"]}
    prior = gaussian.sample_prior(coords=coords)
    curve = gaussian.sample_curve(prior, days=15)

    assert curve.dims == ("chain", "draw", "x", "event")
    assert curve.event.size == 2
    assert curve.x.size == 100


def test_event_effect_different_dims():
    # Test EventEffect with different dimension configurations
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=[7, 5], sigma=1, dims="campaign"),
        },
    )
    effect_size = Prior("Normal", mu=[1, 2], sigma=1, dims="campaign")
    effect = EventEffect(basis=gaussian, effect_size=effect_size, dims=("campaign",))

    # Create test data
    X = np.random.randn(10, 2)  # 10 time points, 2 campaigns
    coords = {"campaign": ["Campaign1", "Campaign2"]}

    with pm.Model(coords=coords) as model:
        effect.apply(X)
        assert "campaign" in model.coords


def test_basis_matrix_creation():
    # Test the creation of basis matrix with different event configurations
    df_events = pd.DataFrame(
        {
            "event": ["event1", "event2", "event3"],
            "start_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
            "end_date": pd.to_datetime(["2023-01-05", "2023-02-05", "2023-03-05"]),
        }
    )

    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    def create_basis_matrix(df_events: pd.DataFrame, model_dates: np.ndarray):
        start_dates = df_events["start_date"]
        end_dates = df_events["end_date"]

        def difference_in_days(model_dates, event_dates):
            if hasattr(model_dates, "to_numpy"):
                model_dates = model_dates.to_numpy()
            if hasattr(event_dates, "to_numpy"):
                event_dates = event_dates.to_numpy()
            return (model_dates[:, None] - event_dates) / np.timedelta64(1, "D")

        s_ref = difference_in_days(model_dates, start_dates)
        e_ref = difference_in_days(model_dates, end_dates)

        return np.where(
            (s_ref >= 0) & (e_ref <= 0),
            0,
            np.where(np.abs(s_ref) < np.abs(e_ref), s_ref, e_ref),
        )

    X = create_basis_matrix(df_events, dates)

    # Test shape
    assert X.shape == (len(dates), len(df_events))

    # Test values during event periods
    event1_mask = (dates >= df_events.loc[0, "start_date"]) & (
        dates <= df_events.loc[0, "end_date"]
    )
    assert_array_equal(X[event1_mask, 0], np.zeros(sum(event1_mask)))


def test_event_effect_serialization_roundtrip():
    # Test complete serialization roundtrip with complex configuration
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=[7, 5, 3], sigma=1, dims="event"),
        },
    )
    effect_size = Prior("Normal", mu=[1, 2, 3], sigma=1, dims="event")
    original_effect = EventEffect(
        basis=gaussian, effect_size=effect_size, dims=("event",)
    )

    # Serialize
    effect_dict = original_effect.to_dict()

    # Deserialize
    restored_effect = EventEffect.from_dict(effect_dict["data"])

    # Compare all attributes
    assert restored_effect.dims == original_effect.dims
    assert restored_effect.basis.lookup_name == original_effect.basis.lookup_name
    assert restored_effect.basis.prefix == original_effect.basis.prefix
    assert (
        restored_effect.effect_size.to_dict() == original_effect.effect_size.to_dict()
    )


def test_gaussian_basis_large_sigma():
    """Test GaussianBasis behavior with very large sigma values."""
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=1000, sigma=1),
        },
    )

    parameters = xr.Dataset(
        {"sigma": (["chain", "draw"], np.array([[1000.0]]))},
        coords={
            "chain": [0],
            "draw": [0],
        },
    )

    curve = gaussian.sample_curve(parameters, days=10)

    # With large sigma, the curve should be very flat
    assert np.allclose(curve[0], curve[-1], rtol=0.1)


def test_basis_matrix_overlapping_events():
    """Test basis matrix creation with overlapping events."""
    df_events = pd.DataFrame(
        {
            "event": ["event1", "event2"],
            "start_date": pd.to_datetime(["2023-01-01", "2023-01-03"]),
            "end_date": pd.to_datetime(["2023-01-05", "2023-01-07"]),
        }
    )

    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    def create_basis_matrix(df_events: pd.DataFrame, model_dates: np.ndarray):
        start_dates = df_events["start_date"]
        end_dates = df_events["end_date"]

        def difference_in_days(model_dates, event_dates):
            if hasattr(model_dates, "to_numpy"):
                model_dates = model_dates.to_numpy()
            if hasattr(event_dates, "to_numpy"):
                event_dates = event_dates.to_numpy()
            return (model_dates[:, None] - event_dates) / np.timedelta64(1, "D")

        s_ref = difference_in_days(model_dates, start_dates)
        e_ref = difference_in_days(model_dates, end_dates)

        return np.where(
            (s_ref >= 0) & (e_ref <= 0),
            0,
            np.where(np.abs(s_ref) < np.abs(e_ref), s_ref, e_ref),
        )

    X = create_basis_matrix(df_events, dates)

    # Check overlapping period
    overlap_mask = (dates >= df_events.loc[1, "start_date"]) & (
        dates <= df_events.loc[0, "end_date"]
    )
    overlap_indices = np.where(overlap_mask)[0]

    # Both events should have effect during overlap
    assert np.all(X[overlap_indices, 0] == 0)
    assert np.all(X[overlap_indices, 1] == 0)


def test_gaussian_basis_symmetry():
    """Test that GaussianBasis produces symmetric curves."""
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=5, sigma=1),
        },
    )

    parameters = xr.Dataset(
        {"sigma": (["chain", "draw"], np.array([[5.0]]))},
        coords={
            "chain": [0],
            "draw": [0],
        },
    )

    curve = gaussian.sample_curve(parameters, days=10)

    # Test symmetry around x=0
    mid_point = len(curve.x) // 2
    assert np.allclose(curve[:mid_point], curve[-1 : -mid_point - 1 : -1])


def test_basis_matrix_edge_dates():
    """Test basis matrix creation with edge case dates."""
    df_events = pd.DataFrame(
        {
            "event": ["event1"],
            "start_date": pd.to_datetime(["2023-01-01"]),
            "end_date": pd.to_datetime(["2023-01-01"]),  # Same day start and end
        }
    )

    dates = pd.date_range("2023-01-01", periods=3, freq="D")

    def create_basis_matrix(df_events: pd.DataFrame, model_dates: np.ndarray):
        start_dates = df_events["start_date"]
        end_dates = df_events["end_date"]

        def difference_in_days(model_dates, event_dates):
            if hasattr(model_dates, "to_numpy"):
                model_dates = model_dates.to_numpy()
            if hasattr(event_dates, "to_numpy"):
                event_dates = event_dates.to_numpy()
            return (model_dates[:, None] - event_dates) / np.timedelta64(1, "D")

        s_ref = difference_in_days(model_dates, start_dates)
        e_ref = difference_in_days(model_dates, end_dates)

        return np.where(
            (s_ref >= 0) & (e_ref <= 0),
            0,
            np.where(np.abs(s_ref) < np.abs(e_ref), s_ref, e_ref),
        )

    X = create_basis_matrix(df_events, dates)

    # Test single-day event
    assert X[0, 0] == 0  # Event day should be 0
    assert X[1, 0] > 0  # Day after should be positive
    assert X[-1, 0] > X[1, 0]  # Effect should increase with distance


@pytest.mark.parametrize(
    "dates_constructor",
    [pd.Series, pd.to_datetime],
    ids=["Series", "DatetimeIndex"],
)
@pytest.mark.parametrize(
    "reference_constructor",
    [str, pd.to_datetime],
    ids=["str", "Timestamp"],
)
def test_days_from_reference(dates_constructor, reference_constructor):
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    dates = dates_constructor(dates)

    reference_date = pd.to_datetime("2023-01-05")
    reference_date = reference_constructor(reference_date)

    result = days_from_reference(
        dates=dates,
        reference_date=reference_date,
    )

    np.testing.assert_allclose(result, np.arange(-4, 6))


@pytest.mark.parametrize("reference_date", ["2000-01-05", "2100-01-10"])
def test_basis_matrix_date_agnostic(reference_date) -> None:
    dates = pd.date_range("2023-01-01", periods=4, freq="D")

    start_dates = pd.to_datetime(["2023-01-01", "2023-01-20"])
    end_dates = pd.to_datetime(["2023-01-02", "2023-01-25"])

    days = days_from_reference(dates, reference_date)
    s_diff = days_from_reference(
        dates=start_dates,
        reference_date=reference_date,
    )
    e_diff = days_from_reference(
        dates=end_dates,
        reference_date=reference_date,
    )

    s_ref = days[:, None] - s_diff
    e_ref = days[:, None] - e_diff

    def create_basis_matrix(s_ref, e_ref):
        return np.where(
            (s_ref >= 0) & (e_ref <= 0),
            0,
            np.where(np.abs(s_ref) < np.abs(e_ref), s_ref, e_ref),
        )

    result = create_basis_matrix(s_ref, e_ref)

    np.testing.assert_array_equal(
        result,
        np.array([[0, -19], [0, -18], [1, -17], [2, -16]]),
    )


@pytest.mark.parametrize(
    "sigma_dims, effect_dims",
    [
        pytest.param("something else", "event", id="basis_not_subset"),
        pytest.param("event", "something else", id="effect_not_subset"),
    ],
)
def test_event_effect_dim_validation(sigma_dims, effect_dims) -> None:
    basis = GaussianBasis(
        priors={
            "sigma": Prior("HalfNormal", dims=sigma_dims),
        }
    )
    effect_size = Prior("Normal", dims=effect_dims)

    with pytest.raises(ValueError):
        EventEffect(basis=basis, effect_size=effect_size, dims="event")
