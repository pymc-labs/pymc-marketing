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
"""Shared choice-data behaviour for ``MNLogit`` and ``NestedLogit``.

These tests exercise only the surface the two classes inherit from
``ChoiceDataMixin``: supplying choice data to ``fit`` and
``sample_prior_predictive``, missing constructor settings, the first-slot type
check, the constructor deprecation warning, the refit guard against mismatched
or in-place edited data, and the save/load round trip. Model-specific
behaviour stays in the per-class test modules.
"""

import json
import re
import warnings
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.customer_choice.mnl_logit import MNLogit
from pymc_marketing.customer_choice.nested_logit import NestedLogit


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "choice": ["alt", "another", "other", "option"],
            "alt_X1": [1, 2, 3, 7],
            "alt_X2": [4, 5, 6, 5],
            "other_X1": [1, 2, 4, 7],
            "other_X2": [5, 6, 8, 0],
            "option_X1": [4, 6, 7, 1],
            "option_X2": [6, 6, 7, 4],
            "another_X1": [5, 3, 7, 8],
            "another_X2": [7, 4, 2, 8],
            "income": [50000, 60000, 70000, 90002],
        }
    )


@pytest.fixture
def utility_eqs():
    return [
        "alt ~ alt_X1 + alt_X2 | income",
        "other ~ other_X1 + other_X2 | income",
        "option ~ option_X1 + option_X2 | income",
        "another ~ another_X1 + another_X2 | income",
    ]


@pytest.fixture
def other_utility_eqs(utility_eqs):
    return list(reversed(utility_eqs))


class _ModelCase:
    """A concrete model class with its required and extra constructor kwargs."""

    def __init__(self, cls, required, extras=None):
        self.cls = cls
        self.required = required
        self.extras = extras or {}

    def build(self, **overrides):
        return self.cls(**{**self.required, **self.extras, **overrides})


@pytest.fixture(
    params=[
        pytest.param("mnl", id="mnl"),
        pytest.param("nested", id="nested"),
        pytest.param("nested_alphas_nests", id="nested-alphas-nests"),
    ]
)
def model_case(request, utility_eqs):
    """Per-class constructor settings, so both classes share one test surface."""
    if request.param == "mnl":
        return _ModelCase(
            MNLogit,
            {
                "utility_equations": utility_eqs,
                "depvar": "choice",
                "covariates": ["X1", "X2"],
            },
        )

    extras = {"alphas_nests": True} if request.param == "nested_alphas_nests" else {}
    return _ModelCase(
        NestedLogit,
        {
            "utility_equations": utility_eqs,
            "depvar": "choice",
            "covariates": ["X1", "X2"],
            "nesting_structure": {
                "nest1": ["alt", "other"],
                "nest2": ["option", "another"],
            },
        },
        extras,
    )


def test_fit_accepts_choice_data(model_case, sample_df, mock_pymc_sample):
    model = model_case.build()

    model.fit(sample_df)

    pd.testing.assert_frame_equal(model.choice_df, sample_df)


def test_sample_accepts_choice_data(model_case, sample_df, mock_pymc_sample):
    model = model_case.build()

    model.sample(sample_prior_predictive_kwargs={"choice_df": sample_df})

    assert "prior_predictive" in model.idata
    assert "posterior_predictive" in model.idata


def test_missing_setting_raises(model_case):
    for name in model_case.required:
        kwargs = {**model_case.required, **model_case.extras}
        kwargs.pop(name)

        with pytest.raises(
            ValueError, match=re.escape(f"Missing required argument(s): {name}.")
        ):
            model_case.cls(**kwargs)


def test_positional_settings_without_data_raise_keyword_error(model_case):
    # New-style call written positionally: the utility equations land in the
    # deprecated data slot, so the error must point at the keyword arguments
    # (and fire before the missing-argument check would report ``covariates``).
    # ``model_case.required`` preserves the constructor's positional order.
    positional = tuple(model_case.required.values())

    with pytest.raises(TypeError) as excinfo:
        model_case.cls(*positional)

    message = str(excinfo.value)
    for name in model_case.required:
        assert name in message


@pytest.mark.parametrize(
    "bad_choice_df",
    [
        pytest.param({}, id="dict"),
        pytest.param(np.array([1, 2, 3]), id="array"),
    ],
)
def test_non_dataframe_first_positional_raises(model_case, bad_choice_df):
    with pytest.raises(TypeError, match="keyword arguments"):
        model_case.cls(bad_choice_df, **model_case.required, **model_case.extras)


def test_deprecated_choice_df_warns(model_case, sample_df):
    # Build positionally, in the real constructor order, so the warning's
    # stacklevel=3 must point at this test file (the user call site).
    positional = (sample_df, *model_case.required.values())
    with pytest.warns(
        DeprecationWarning, match="removed in a future release"
    ) as record:
        model = model_case.cls(*positional)

    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert deprecations
    assert Path(deprecations[0].filename).resolve() == Path(__file__).resolve()

    assert model.choice_df.equals(sample_df)
    model.build_model()
    assert hasattr(model, "model")


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("build_model", {}),
        ("sample_prior_predictive", {}),
        ("sample_posterior_predictive", {}),
        ("create_fit_data_group", {}),
        ("sample", {}),
        ("apply_intervention", {"new_choice_df": pd.DataFrame()}),
    ],
)
def test_data_dependent_methods_require_choice_data(model_case, method, kwargs):
    model = model_case.build()

    with pytest.raises(ValueError, match="Choice data is required"):
        getattr(model, method)(**kwargs)


def test_refit_with_different_data_raises(model_case, sample_df, mock_pymc_sample):
    model = model_case.build()
    model.fit(sample_df)

    changed_df = sample_df.copy()
    changed_df["alt_X1"] = changed_df["alt_X1"] * 1.2

    with pytest.raises(ValueError, match="built with different choice data"):
        model.fit(changed_df)

    model.fit(sample_df)  # same data is fine


def test_refit_with_different_utility_equations_raises(
    model_case, sample_df, other_utility_eqs, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)

    with pytest.raises(ValueError, match="different utility equations"):
        model.fit(utility_equations=other_utility_eqs)


def test_refit_with_fresh_copy_of_same_data(model_case, sample_df, mock_pymc_sample):
    model = model_case.build()
    fresh_df = sample_df.copy()
    model.fit(sample_df)

    assert list(sample_df.columns) == list(fresh_df.columns)
    model.fit(fresh_df)


def test_in_place_edit_then_refit_raises(model_case, sample_df, mock_pymc_sample):
    # The model stores a copy, so mutating the caller's frame after fitting must
    # make the refit reject the now-mismatched data.
    model = model_case.build()
    model.fit(sample_df)

    sample_df.loc[0, "alt_X1"] = sample_df.loc[0, "alt_X1"] + 1

    with pytest.raises(ValueError, match="built with different choice data"):
        model.fit(sample_df)


def test_prior_predictive_different_data_raises(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()
    model.sample_prior_predictive(choice_df=sample_df)

    changed_df = sample_df.copy()
    changed_df["alt_X1"] = changed_df["alt_X1"] * 1.2

    with pytest.raises(ValueError, match="built with different choice data"):
        model.sample_prior_predictive(choice_df=changed_df)


def test_prior_predictive_different_equations_raises(
    model_case, sample_df, other_utility_eqs, mock_pymc_sample
):
    model = model_case.build()
    model.sample_prior_predictive(choice_df=sample_df)

    with pytest.raises(ValueError, match="different utility equations"):
        model.sample_prior_predictive(utility_equations=other_utility_eqs)


def test_fit_prior_predictive_then_fit_keeps_stored_data(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()

    model.fit(sample_df)
    model.sample_prior_predictive()
    model.fit()

    pd.testing.assert_frame_equal(model.choice_df, sample_df)


def test_loaded_model_refit_guards_on_data(
    model_case, sample_df, tmp_path, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df, random_seed=42)

    path = tmp_path / "model.nc"
    model.save(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        loaded = model_case.cls.load(str(path))

    # The raises checks run against the frame restored from NetCDF. Exact
    # comparison means an edit below the default relative tolerance is still
    # treated as different data.
    tiny_edit_df = sample_df.copy()
    tiny_edit_df["alt_X1"] = tiny_edit_df["alt_X1"] + 1e-9
    with pytest.raises(ValueError, match="built with different choice data"):
        loaded.fit(tiny_edit_df)

    changed_df = sample_df.copy()
    changed_df["alt_X1"] = changed_df["alt_X1"] * 1.2
    with pytest.raises(ValueError, match="built with different choice data"):
        loaded.fit(changed_df)

    # A refit on the original data survives the NetCDF dtype round trip.
    loaded.fit(sample_df)

    # The save/load round trip drops the index name, so a frame that names its
    # index must still be accepted.
    named_df = sample_df.copy()
    named_df.index.name = "obs"
    loaded.fit(named_df)


def test_refit_with_in_place_edited_utility_equations_raises(
    model_case, sample_df, utility_eqs, mock_pymc_sample
):
    model = model_case.build()
    eqs = list(utility_eqs)
    model.fit(sample_df, utility_equations=eqs)

    eqs[0] = eqs[0] + " + alt_X1"

    with pytest.raises(ValueError, match="different utility equations"):
        model.fit(utility_equations=eqs)


def test_save_load_roundtrip(model_case, sample_df, tmp_path, mock_pymc_sample):
    model = model_case.build()
    model.fit(sample_df, random_seed=42)

    path = tmp_path / "model.nc"
    model.save(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        loaded = model_case.cls.load(str(path))

    # Constructor settings are rebuilt from the serialised attrs.
    for name, value in {**model_case.required, **model_case.extras}.items():
        assert getattr(loaded, name) == value
    assert loaded.model_config == model.model_config

    # Choice data is restored from the fit_data group with its index and dtypes.
    pd.testing.assert_frame_equal(
        loaded.choice_df, model.choice_df, check_index_type=False
    )

    # The choice data is not duplicated into the attrs as a placeholder.
    assert json.loads(model.idata.attrs["choice_df"]) is None

    # Posterior and model id survive the file round trip.
    xr.testing.assert_equal(loaded.idata["posterior"], model.idata["posterior"])
    assert loaded.id == model.id

    # Posterior predictive sampling runs on the loaded model.
    post_pred = loaded.sample_posterior_predictive(extend_idata=False)
    assert "posterior_predictive" in post_pred

    if isinstance(loaded, NestedLogit):
        expected_alphas_nests = model_case.extras.get("alphas_nests", False)
        assert loaded.alphas_nests == expected_alphas_nests
        assert ("alphas_nests" in loaded.model.named_vars) == expected_alphas_nests


@pytest.fixture(params=["appearance-order", "reordered-with-unused"])
def categorical_df(request, sample_df):
    if request.param == "appearance-order":
        return sample_df.assign(choice=pd.Categorical(sample_df["choice"]))
    # Category order differs from the utility equations and "unused" has no rows.
    return sample_df.assign(
        choice=pd.Categorical(
            sample_df["choice"],
            categories=["unused", "another", "option", "other", "alt"],
        )
    )


@pytest.fixture
def ordered_int_df(sample_df):
    values = [1 + (i % 4) for i in range(len(sample_df))]
    values[1] = None
    return sample_df.assign(
        rating=pd.Categorical(values, categories=[1, 2, 3, 4], ordered=True)
    )


def test_save_load_categorical_depvar(
    model_case, categorical_df, tmp_path, mock_pymc_sample
):
    model = model_case.build()
    model.fit(categorical_df, random_seed=42)

    path = tmp_path / "model.nc"
    model.save(str(path))
    loaded = model_case.cls.load(str(path))

    # The model build inside load() is the real guard.
    # The `alts` coords alone are weak: they come from the utility equations, not the data.
    assert isinstance(loaded.choice_df["choice"].dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(
        loaded.choice_df, model.choice_df, check_index_type=False
    )

    assert list(loaded.model.coords["alts"]) == list(model.model.coords["alts"])
    assert loaded.alternatives == model.alternatives

    # The loaded model accepts the original categorical data without raising.
    loaded.fit(categorical_df, random_seed=42)


def test_save_load_ordered_non_string_categorical(
    model_case, ordered_int_df, tmp_path, mock_pymc_sample
):
    model = model_case.build()
    model.fit(ordered_int_df, random_seed=42)

    path = tmp_path / "model.nc"
    model.save(str(path))
    loaded = model_case.cls.load(str(path))

    restored = loaded.choice_df["rating"]
    assert isinstance(restored.dtype, pd.CategoricalDtype)
    assert restored.cat.ordered
    assert list(restored.cat.categories) == [1, 2, 3, 4]
    # The NaN stays NaN while the non-missing codes round-trip (no all-NaN bug).
    assert restored.isna().tolist() == ordered_int_df["rating"].isna().tolist()
    assert restored.notna().any()
    pd.testing.assert_series_equal(restored, model.choice_df["rating"])


def test_custom_betas_fixed_prior_survives_save_load(
    model_case, sample_df, tmp_path, mock_pymc_sample
):
    custom_prior = Prior("Normal", mu=1.0, sigma=2.0, dims=("alts", "fixed_covariates"))
    model = model_case.build(model_config={"betas_fixed_": custom_prior})
    model.fit(sample_df, random_seed=42)

    path = tmp_path / "model.nc"
    model.save(str(path))
    loaded = model_case.cls.load(str(path))

    # Invariant: the custom `betas_fixed_` prior survives save/load and the id check passes.
    # The prior comparison is the real guard; id equality is a sanity check.
    assert loaded.model_config["betas_fixed_"].to_dict() == custom_prior.to_dict()
    assert loaded.id == model.id


def _expected_preprocessed(model_case, df, utility_equations=None):
    """The arrays a freshly built instance preprocesses ``df`` into."""
    reference = model_case.build()
    equations = (
        reference.utility_equations if utility_equations is None else utility_equations
    )
    return reference.preprocess_model_data(df, equations)


def _assert_model_data(model, X, F, y):
    """Assert the fitted model's data containers hold the given arrays."""
    np.testing.assert_array_equal(model.model["X"].get_value(), X)
    np.testing.assert_array_equal(model.model["y"].get_value(), y)
    if F is not None and len(F) > 0:
        np.testing.assert_array_equal(
            model.model[model._fixed_data_name].get_value(), F
        )


def _instance_snapshot(model):
    """Copy the instance state a restore must put back."""
    X, F, y = model._training_arrays()
    snapshot = {
        "X": X.copy(),
        "F": None if F is None else F.copy(),
        "y": y.copy(),
        "alternatives": list(model.alternatives),
        "coords": {name: list(values) for name, values in model.coords.items()},
    }
    if hasattr(model, "nest_indices"):
        snapshot["nest_indices"] = {
            name: list(indices) for name, indices in model.nest_indices["top"].items()
        }
        snapshot["all_nests"] = list(model.all_nests)
        snapshot["lambda_lkup"] = dict(model.lambda_lkup)
    return snapshot


def _assert_instance_state(model, snapshot):
    """Assert the instance state still matches a snapshot taken earlier."""
    X, F, y = model._training_arrays()
    np.testing.assert_array_equal(X, snapshot["X"])
    if snapshot["F"] is None:
        assert F is None
    else:
        np.testing.assert_array_equal(F, snapshot["F"])
    np.testing.assert_array_equal(y, snapshot["y"])
    assert list(model.alternatives) == snapshot["alternatives"]
    assert {name: list(values) for name, values in model.coords.items()} == snapshot[
        "coords"
    ]
    if hasattr(model, "nest_indices"):
        assert {
            name: list(indices) for name, indices in model.nest_indices["top"].items()
        } == snapshot["nest_indices"]
        assert list(model.all_nests) == snapshot["all_nests"]
        assert dict(model.lambda_lkup) == snapshot["lambda_lkup"]


def _copy_attribute(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_attribute(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_attribute(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_attribute(item) for item in value)
    return value


def _assert_attribute_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_attribute_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_attribute_equal(actual_item, expected_item)
    else:
        assert actual is expected or actual == expected


def _attribute_snapshot(instance):
    """Snapshot every instance attribute, copying mutable state."""
    return {name: _copy_attribute(value) for name, value in vars(instance).items()}


def _assert_attributes_unchanged(instance, snapshot):
    assert vars(instance).keys() == snapshot.keys()
    for name, expected in snapshot.items():
        _assert_attribute_equal(vars(instance)[name], expected)


def test_private_preprocess_preserves_instance_state(
    model_case, sample_df, other_utility_eqs
):
    model = model_case.build()
    model._set_choice_inputs(sample_df)
    model.build_model()
    snapshot = _attribute_snapshot(model)

    # Different rows and equations, so re-assigning any stored state would
    # change its content and fail the snapshot comparison below.
    other_df = pd.concat([sample_df, sample_df], ignore_index=True)
    bundle = model._preprocess(other_df, other_utility_eqs)

    assert is_dataclass(bundle)
    assert bundle.__dataclass_params__.frozen
    assert bundle.X.shape[0] == len(other_df)
    assert bundle.y.shape == (len(other_df),)
    with pytest.raises(FrozenInstanceError):
        bundle.X = bundle.X
    _assert_attributes_unchanged(model, snapshot)


def test_preprocess_model_data_preserves_instance_state(
    model_case, sample_df, other_utility_eqs
):
    model = model_case.build()
    model._set_choice_inputs(sample_df)
    model.build_model()
    snapshot = _attribute_snapshot(model)

    # Different rows and equations, so re-assigning any stored state would
    # change its content and fail the snapshot comparison below.
    other_df = pd.concat([sample_df, sample_df], ignore_index=True)
    X, _F, y = model.preprocess_model_data(other_df, other_utility_eqs)

    assert X.shape[0] == len(other_df)
    assert y.shape == (len(other_df),)
    _assert_attributes_unchanged(model, snapshot)


def test_build_model_stores_preprocessed_state(model_case, sample_df):
    model = model_case.build()
    stored_names = {"alternatives", "coords", "y", "F", model._X_data_name}
    assert stored_names.isdisjoint(vars(model))

    bundle = model._preprocess(sample_df, model.utility_equations)
    assert stored_names.isdisjoint(vars(model))

    model._set_choice_inputs(sample_df)
    model.build_model()

    assert stored_names <= set(vars(model))
    X, F, y = model._training_arrays()
    assert getattr(model, model._X_data_name) is X
    np.testing.assert_array_equal(X, bundle.X)
    np.testing.assert_array_equal(y, bundle.y)
    if bundle.F is None:
        assert F is None
    else:
        np.testing.assert_array_equal(F, bundle.F)
    assert model.alternatives == bundle.alternatives
    assert {name: list(values) for name, values in model.coords.items()} == {
        name: list(values) for name, values in bundle.coords.items()
    }


def test_make_model_uses_coords_override(model_case, sample_df):
    model = model_case.build()
    model._set_choice_inputs(sample_df)
    model.build_model()
    bundle = model._preprocess(sample_df, model.utility_equations)
    overridden_alts = list(reversed(bundle.coords["alts"]))
    coords = {**bundle.coords, "alts": overridden_alts}
    stored_coords = {name: list(values) for name, values in model.coords.items()}

    built = model.make_model(bundle.X, bundle.F, bundle.y, coords=coords)

    assert list(built.coords["alts"]) == overridden_alts
    assert list(built.coords["alts"]) != list(bundle.coords["alts"])
    assert {
        name: list(values) for name, values in model.coords.items()
    } == stored_coords


def test_refit_after_intervention_uses_training_data(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)

    intervened = sample_df.copy()
    intervened["alt_X1"] = intervened["alt_X1"] + 100.0
    model.apply_intervention(intervened)

    model.fit(sample_df)

    _assert_model_data(model, expected_X, expected_F, expected_y)


def test_refit_after_posterior_predictive_uses_training_data(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)

    intervened = sample_df.copy()
    intervened["alt_X1"] = intervened["alt_X1"] + 100.0
    model.sample_posterior_predictive(choice_df=intervened)

    model.fit(sample_df)

    _assert_model_data(model, expected_X, expected_F, expected_y)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("apply_intervention", id="apply-intervention"),
        pytest.param("sample_posterior_predictive", id="sample-posterior-predictive"),
    ],
)
def test_refit_after_different_row_count_uses_training_data(
    model_case, sample_df, mock_pymc_sample, method
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)

    intervened = pd.concat([sample_df, sample_df], ignore_index=True)
    if method == "apply_intervention":
        model.apply_intervention(intervened)
    else:
        model.sample_posterior_predictive(choice_df=intervened)

    model.fit(sample_df)

    _assert_model_data(model, expected_X, expected_F, expected_y)
    assert model.model["y"].get_value().shape == (len(sample_df),)
    assert len(model.model.coords["obs"]) == len(sample_df)


def test_sample_posterior_predictive_after_intervention_uses_stored_data(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)
    snapshot = _instance_snapshot(model)

    intervened = pd.concat([sample_df, sample_df], ignore_index=True)
    intervened["alt_X1"] = intervened["alt_X1"] + 100.0
    model.apply_intervention(intervened)

    post_pred = model.sample_posterior_predictive()

    assert post_pred["posterior_predictive"]["p"].sizes["obs"] == len(sample_df)
    _assert_model_data(model, expected_X, expected_F, expected_y)
    _assert_instance_state(model, snapshot)


def test_intervention_with_different_row_count_restores_model(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)

    intervened = pd.concat([sample_df, sample_df], ignore_index=True)
    model.apply_intervention(intervened)

    assert model.model["X"].get_value().shape[0] == len(sample_df)
    _assert_model_data(model, expected_X, expected_F, expected_y)
    assert len(model.model.coords["obs"]) == len(sample_df)


def test_posterior_predictive_with_different_row_count_restores_model(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)

    intervened = pd.concat([sample_df, sample_df], ignore_index=True)
    model.sample_posterior_predictive(choice_df=intervened)

    _assert_model_data(model, expected_X, expected_F, expected_y)
    assert len(model.model.coords["obs"]) == len(sample_df)


def test_failed_posterior_predictive_restores_model(
    model_case, sample_df, monkeypatch, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)
    snapshot = _instance_snapshot(model)

    intervened = pd.concat([sample_df, sample_df], ignore_index=True)

    def boom(*args, **kwargs):
        raise RuntimeError("sampling blew up")

    monkeypatch.setattr(pm, "sample_posterior_predictive", boom)

    with pytest.raises(RuntimeError, match="sampling blew up"):
        model.sample_posterior_predictive(choice_df=intervened)

    _assert_model_data(model, expected_X, expected_F, expected_y)
    _assert_instance_state(model, snapshot)
    assert len(model.model.coords["obs"]) == len(sample_df)


def test_failed_reset_adds_note_to_original_error(
    model_case, sample_df, monkeypatch, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)

    intervened = pd.concat([sample_df, sample_df], ignore_index=True)

    def sample_boom(*args, **kwargs):
        raise RuntimeError("sampling blew up")

    def reset_boom():
        raise RuntimeError("reset failed")

    monkeypatch.setattr(pm, "sample_posterior_predictive", sample_boom)
    monkeypatch.setattr(model, "_training_arrays", reset_boom)

    with pytest.raises(RuntimeError, match="sampling blew up") as excinfo:
        model.sample_posterior_predictive(choice_df=intervened)

    assert excinfo.value.__cause__ is None
    assert any("reset failed" in note for note in excinfo.value.__notes__)


def _forbid_model_data_swap(model, monkeypatch):
    """Fail if a no-data prediction preprocesses data or swaps it into the model."""

    def fail(*args, **kwargs):
        raise AssertionError("a no-data prediction must not touch model data")

    for name in ("preprocess_model_data", "_preprocess", "_set_model_data"):
        monkeypatch.setattr(model, name, fail)


def test_default_posterior_predictive_uses_training_data(
    model_case, sample_df, monkeypatch, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)
    snapshot = _instance_snapshot(model)
    _forbid_model_data_swap(model, monkeypatch)

    post_pred = model.sample_posterior_predictive()

    assert post_pred["posterior_predictive"]["p"].sizes["obs"] == len(sample_df)
    _assert_model_data(model, expected_X, expected_F, expected_y)
    _assert_instance_state(model, snapshot)


def test_intervention_without_new_data_uses_training_data(
    model_case, sample_df, monkeypatch, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)
    snapshot = _instance_snapshot(model)
    _forbid_model_data_swap(model, monkeypatch)

    intervention = model.apply_intervention(None)

    assert intervention["posterior_predictive"]["p"].sizes["obs"] == len(sample_df)
    _assert_model_data(model, expected_X, expected_F, expected_y)
    _assert_instance_state(model, snapshot)


def test_failed_data_swap_restores_training_data(
    model_case, sample_df, monkeypatch, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(model_case, sample_df)
    snapshot = _instance_snapshot(model)
    intervened = pd.concat([sample_df, sample_df], ignore_index=True)

    set_model_data = model._set_model_data

    def swap_X_and_y_then_fail(X, F, y):
        if len(X) == len(sample_df):
            set_model_data(X, F, y)
            return
        # X and y now hold the prediction data, while the fixed covariates are
        # rejected before they are set.
        set_model_data(X, None, y)
        raise ValueError("fixed covariates have an incompatible shape")

    monkeypatch.setattr(model, "_set_model_data", swap_X_and_y_then_fail)

    with pytest.raises(ValueError, match="incompatible shape"):
        model.sample_posterior_predictive(choice_df=intervened)

    _assert_model_data(model, expected_X, expected_F, expected_y)
    _assert_instance_state(model, snapshot)
    assert len(model.model.coords["obs"]) == len(sample_df)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("apply_intervention", id="apply-intervention"),
        pytest.param("sample_posterior_predictive", id="sample-posterior-predictive"),
    ],
)
def test_no_fixed_covariates_different_row_count(
    model_case, sample_df, utility_eqs, mock_pymc_sample, method
):
    no_fixed = [equation.split(" | ")[0] for equation in utility_eqs]
    model = model_case.build(utility_equations=no_fixed)
    model.fit(sample_df)
    expected_X, expected_F, expected_y = _expected_preprocessed(
        model_case, sample_df, no_fixed
    )
    assert "F" not in model.model.named_vars
    assert "W" not in model.model.named_vars

    intervened = pd.concat([sample_df, sample_df], ignore_index=True)
    if method == "apply_intervention":
        model.apply_intervention(intervened)
    else:
        model.sample_posterior_predictive(choice_df=intervened)

    assert model.model["X"].get_value().shape[0] == len(sample_df)
    _assert_model_data(model, expected_X, expected_F, expected_y)
    assert len(model.model.coords["obs"]) == len(sample_df)


def test_new_equations_intervention_keeps_fitted_model(
    model_case, sample_df, utility_eqs, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)
    fitted_model = model.model
    snapshot = _instance_snapshot(model)
    fitted_X = fitted_model["X"].get_value().copy()
    fitted_y = fitted_model["y"].get_value().copy()

    # New equations reorder the alternatives and the new data has more rows.
    new_equations = list(reversed(utility_eqs))
    intervened = pd.concat([sample_df, sample_df], ignore_index=True)
    model.apply_intervention(intervened, new_equations)

    assert model.model is fitted_model
    np.testing.assert_array_equal(fitted_model["X"].get_value(), fitted_X)
    np.testing.assert_array_equal(fitted_model["y"].get_value(), fitted_y)
    _assert_instance_state(model, snapshot)


def test_predictive_methods_return_types_unchanged(
    model_case, sample_df, mock_pymc_sample
):
    model = model_case.build()
    model.fit(sample_df)

    post_pred = model.sample_posterior_predictive(
        choice_df=sample_df, extend_idata=False
    )
    assert isinstance(post_pred, xr.DataTree)
    assert "posterior_predictive" in post_pred

    intervention = model.apply_intervention(sample_df.copy())
    assert isinstance(intervention, xr.DataTree)
    assert "posterior_predictive" in intervention
    assert model.intervention_idata is intervention
