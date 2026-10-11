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
``ChoiceDataMixin``: supplying choice data to ``fit``, missing constructor
settings, the constructor deprecation warning and the save/load round trip.
Model-specific behaviour stays in the per-class test modules.
"""

import json
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

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

    with pytest.raises(ValueError, match="different data"):
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
