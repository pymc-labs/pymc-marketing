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
"""Tests for pymc_marketing.mmm.pie."""

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior
from scipy.stats import pearsonr

from pymc_marketing.mmm.pie import PIEModel, generate_synthetic_rct_corpus

EXPECTED_COLUMNS = [
    "campaign_id",
    "objective",
    "vertical",
    "audience_type",
    "budget",
    "exposure_rate",
    "ctr",
    "avg_treated_outcome",
    "last_click_conversions_per_dollar",
    "measured_incrementality_per_dollar",
]

PRE = ["objective", "vertical", "budget", "audience_type"]
POST = [
    "exposure_rate",
    "ctr",
    "last_click_conversions_per_dollar",
    "avg_treated_outcome",
]


# ---------------------------------------------------------------------------
# generate_synthetic_rct_corpus tests
# ---------------------------------------------------------------------------


def test_synthetic_corpus_schema():
    df = generate_synthetic_rct_corpus(n_campaigns=50, seed=0)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 50

    for col in ("objective", "vertical", "audience_type"):
        assert df[col].dtype == object, f"{col} should be object dtype"

    assert (df["budget"] >= 1_000).all() and (df["budget"] <= 100_000).all()
    assert (df["exposure_rate"] >= 0.1).all() and (df["exposure_rate"] <= 0.9).all()
    assert (df["ctr"] >= 0.01).all() and (df["ctr"] <= 0.1).all()
    assert (df["avg_treated_outcome"] >= 0.0).all() and (
        df["avg_treated_outcome"] <= 5.0
    ).all()
    assert (df["last_click_conversions_per_dollar"] >= 0.0).all()

    assert set(df["objective"]).issubset({"conversions", "traffic", "awareness"})
    assert set(df["vertical"]).issubset({"retail", "travel", "finance"})
    assert set(df["audience_type"]).issubset({"prospecting", "retargeting"})


def test_synthetic_corpus_reproducible():
    df1 = generate_synthetic_rct_corpus(n_campaigns=20, seed=7)
    df2 = generate_synthetic_rct_corpus(n_campaigns=20, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_corpus_different_seeds():
    df1 = generate_synthetic_rct_corpus(n_campaigns=20, seed=1)
    df2 = generate_synthetic_rct_corpus(n_campaigns=20, seed=2)
    assert not df1["measured_incrementality_per_dollar"].equals(
        df2["measured_incrementality_per_dollar"]
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_corpus():
    df = generate_synthetic_rct_corpus(n_campaigns=50, seed=42)
    X = df.drop(columns=["campaign_id", "measured_incrementality_per_dollar"])
    y = df["measured_incrementality_per_dollar"]
    return X, y


@pytest.fixture
def default_model():
    """Fresh PIEModel per test — fixture is function-scoped because tests
    mutate the instance via ``build_model``."""
    return PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
    )


# ---------------------------------------------------------------------------
# build_model tests
# ---------------------------------------------------------------------------


def test_build_model(default_model, small_corpus):
    X, y = small_corpus
    default_model.build_model(X, y)

    model = default_model.model
    var_names = {
        v.name for v in model.free_RVs + model.observed_RVs + model.deterministics
    }

    assert any("bart" in n for n in var_names), "Expected 'bart' variable"
    assert any("sigma" in n for n in var_names), "Expected 'sigma' variable"
    assert any("y" in n for n in var_names), "Expected 'y' observed"

    assert "obs" in model.coords
    assert "feature" in model.coords
    assert list(model.coords["obs"]) == X.index.tolist()
    assert list(model.coords["feature"]) == X.columns.tolist()


def test_dtype_categorical_detection(default_model, small_corpus):
    X, y = small_corpus
    default_model.build_model(X, y)

    for col in ("objective", "vertical", "audience_type"):
        assert col in default_model._encoders, f"Expected encoder for {col}"

    for col in ("budget", "exposure_rate", "ctr"):
        assert col not in default_model._encoders, f"Unexpected encoder for {col}"


def test_string_dtype_feature_is_encoded(default_model, small_corpus):
    """Pandas ``StringDtype`` columns (e.g. from ``df.convert_dtypes()`` or a
    pyarrow CSV backend) must be detected as categorical and label-encoded,
    not passed through unchanged to ``X.values.astype(float)``."""
    X, y = small_corpus
    X_string = X.astype({"objective": "string"})
    assert isinstance(X_string["objective"].dtype, pd.StringDtype)

    default_model.build_model(X_string, y)

    assert "objective" in default_model._encoders, (
        "string-dtype column should have been label-encoded"
    )
    # Round-trips through _data_setter without raising.
    default_model._data_setter(X_string, y=None)


def test_model_config_override(small_corpus):
    X, y = small_corpus
    custom = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 50, "alpha": 0.9, "beta": 1.5},
            "sigma": Prior("HalfNormal", sigma=2.0),
        },
    )
    custom.build_model(X, y)
    assert custom.model_config["bart"]["m"] == 50
    assert custom.model_config["sigma"].parameters["sigma"] == 2.0


def test_feature_not_in_X_raises(small_corpus):
    X, y = small_corpus
    bad_model = PIEModel(
        pre_determined_features=["objective", "MISSING_FEATURE"],
        post_determined_features=POST,
    )
    with pytest.raises(ValueError, match="MISSING_FEATURE"):
        bad_model.build_model(X, y)


def test_negative_incrementality_is_allowed(default_model, small_corpus):
    X, y = small_corpus
    y_with_negative_values = y - y.mean()
    default_model.build_model(X, y_with_negative_values)


def test_pymc_bart_missing_raises(small_corpus, monkeypatch):
    import pymc_marketing.mmm.pie as pie_module

    monkeypatch.setattr(pie_module, "pmb", None)

    model = PIEModel(pre_determined_features=PRE, post_determined_features=POST)
    X, y = small_corpus
    with pytest.raises(ImportError, match="pymc-marketing\\[pie\\]"):
        model.build_model(X, y)


# ---------------------------------------------------------------------------
# _data_setter tests
# ---------------------------------------------------------------------------


def test_unseen_categorical_raises(default_model, small_corpus):
    X, y = small_corpus
    default_model.build_model(X, y)

    X_bad = X.copy()
    X_bad.loc[X_bad.index[0], "objective"] = "UNSEEN_LEVEL"

    with pytest.raises(ValueError, match="objective"):
        default_model._data_setter(X_bad)


def test_data_setter_dummy_y_when_none(default_model, small_corpus):
    """When ``y=None`` (predict mode), ``_data_setter`` installs zeros for the
    likelihood's observed variable."""
    X, y = small_corpus
    default_model.build_model(X, y)
    default_model._data_setter(X, y=None)
    y_obs_value = default_model.model["y_obs"].get_value()
    np.testing.assert_array_equal(y_obs_value, np.zeros(len(X)))


def test_data_setter_requires_training_feature_schema(default_model, small_corpus):
    X, y = small_corpus
    default_model.build_model(X, y)

    with pytest.raises(ValueError, match=r"missing.*budget"):
        default_model._data_setter(X.drop(columns=["budget"]))

    X_extra = X.assign(extra_feature=1.0)
    with pytest.raises(ValueError, match=r"extra.*extra_feature"):
        default_model._data_setter(X_extra)


def test_data_setter_reorders_columns_to_training_order(default_model, small_corpus):
    X, y = small_corpus
    default_model.build_model(X, y)

    default_model._data_setter(X[list(reversed(X.columns))], y=None)


# ---------------------------------------------------------------------------
# sample_posterior_predictive tests
# ---------------------------------------------------------------------------


def test_sample_posterior_predictive_unscaled(small_corpus):
    """Draws must be returned in the original (un-scaled) target range.

    We scale y by 1000 so the contrast between scaled (BART output is O(1) and
    ``sigma=HalfNormal(1)`` keeps draws bounded near 1) and unscaled
    (multiplied by ``_target_scale=max|y|`` ≈ 1000) is unambiguous regardless
    of sampler noise from the tiny BART (m=10) and short chain.
    """
    X, y = small_corpus
    y_big = y * 1000.0
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
        },
    )
    model.fit(X, y_big, draws=10, tune=5, chains=1, random_seed=0)

    preds = model.sample_posterior_predictive(X)

    pred_max = float(abs(preds["y"]).max())
    stored_max = float(abs(model.idata.posterior_predictive["y"]).max())
    # Without the un-scale step, predictions would be O(1) (BART output ~1,
    # sigma ~1); with it they should land in O(y_big) ≈ O(1000).
    assert pred_max > 100, f"Predictions look scaled: pred_max={pred_max:.4f}"
    assert stored_max > 100, (
        f"Stored idata predictions look scaled: stored_max={stored_max:.4f}"
    )


def test_sample_posterior_predictive_supports_new_obs_count(small_corpus):
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
        },
    )
    model.fit(X, y, draws=10, tune=5, chains=1, random_seed=0)

    X_new = X.iloc[:7].copy()
    preds = model.sample_posterior_predictive(X_new, extend_idata=False)

    assert preds["y"].sizes["obs"] == len(X_new)
    assert list(preds["y"].coords["obs"].values) == X_new.index.tolist()


# ---------------------------------------------------------------------------
# save/load roundtrip test
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(small_corpus, tmp_path):
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
        },
    )
    model.fit(X, y, draws=10, tune=5, chains=1, random_seed=0)

    fname = str(tmp_path / "pie_model")
    model.save(fname)

    loaded = PIEModel.load(fname)

    assert loaded.pre_determined_features == PRE
    assert loaded.post_determined_features == POST
    assert loaded.target_column == "y"

    assert abs(loaded._target_scale - model._target_scale) < 1e-6

    preds = loaded.sample_posterior_predictive(X)
    assert preds["y"].sizes["obs"] == len(X)


# ---------------------------------------------------------------------------
# predict + partial-config tests
# ---------------------------------------------------------------------------


def test_predict_returns_unscaled_means(small_corpus):
    """``predict`` (inherited) routes through the overridden
    ``sample_posterior_predictive`` and therefore must return values in the
    original target scale, not the BART-internal scaled units."""
    X, y = small_corpus
    y_big = y * 1000.0
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
        },
    )
    model.fit(X, y_big, draws=10, tune=5, chains=1, random_seed=0)

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    # Same un-scale check as for sample_posterior_predictive: scaled means
    # would be O(1); unscaled means should be O(1000).
    assert abs(preds).max() > 100, (
        f"predict() looks scaled: max(|preds|)={abs(preds).max():.4f}"
    )


def test_partial_model_config_override(small_corpus):
    """Top-level keys missing from ``model_config`` fall back to defaults.

    Nested dicts (e.g. ``"bart"``) are replaced wholesale by ``ModelBuilder``'s
    top-level dict merge, but top-level keys (``sigma``, ``categorical_split``)
    that the caller omits should pick up :py:meth:`default_model_config`.
    """
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        # Override only "bart"; omit "sigma" and "categorical_split".
        model_config={"bart": {"m": 25, "alpha": 0.95, "beta": 2.0}},
    )
    assert model.model_config["bart"]["m"] == 25

    # sigma falls back to default Prior("HalfNormal", sigma=1.0)
    assert isinstance(model.model_config["sigma"], Prior)
    assert model.model_config["sigma"].distribution == "HalfNormal"
    assert model.model_config["sigma"].parameters["sigma"] == 1.0

    # categorical_split falls back to "onehot"
    assert model.model_config["categorical_split"] == "onehot"

    # And the partially-configured model actually builds without errors.
    model.build_model(X, y)


# ---------------------------------------------------------------------------
# Slow recovery test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_recovers_known_effects():
    """PIE predictions must correlate with held-out y above r=0.5.

    This is a smoke test, not a benchmark. Uses a small model (m=100) and
    few draws to stay within CI time limits (run with --run-slow).
    """
    df = generate_synthetic_rct_corpus(n_campaigns=200, seed=0)
    split = 150
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train.drop(columns=["campaign_id", "measured_incrementality_per_dollar"])
    y_train = train["measured_incrementality_per_dollar"]
    X_test = test.drop(columns=["campaign_id", "measured_incrementality_per_dollar"])
    y_test = test["measured_incrementality_per_dollar"].values

    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 100, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
        },
    )
    model.fit(X_train, y_train, draws=200, tune=100, chains=1, random_seed=0)

    preds = model.sample_posterior_predictive(X_test, extend_idata=False)
    pred_mean = preds["y"].mean(dim="sample").values

    r, _ = pearsonr(pred_mean, y_test)
    assert r > 0.5, f"Expected Pearson r > 0.5, got r={r:.3f}"
