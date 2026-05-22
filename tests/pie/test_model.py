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
"""Tests for the PIE model."""

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior
from scipy.stats import pearsonr

from pymc_marketing.pie import PIEModel

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


def generate_synthetic_rct_corpus(
    n_campaigns: int = 500,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic RCT corpus (linear DGP) for exercising PIEModel."""
    rng = np.random.default_rng(seed)

    objectives = np.array(["conversions", "traffic", "awareness"])
    verticals = np.array(["retail", "travel", "finance"])
    audience_types = np.array(["prospecting", "retargeting"])

    objective = rng.choice(objectives, size=n_campaigns)
    vertical = rng.choice(verticals, size=n_campaigns)
    audience_type = rng.choice(audience_types, size=n_campaigns)

    budget = rng.uniform(1_000, 100_000, size=n_campaigns)
    exposure_rate = rng.uniform(0.1, 0.9, size=n_campaigns)
    ctr = rng.uniform(0.01, 0.1, size=n_campaigns)
    avg_treated_outcome = rng.uniform(0.0, 5.0, size=n_campaigns)

    obj_oh = (objective[:, None] == objectives).astype(float)
    vert_oh = (vertical[:, None] == verticals).astype(float)
    aud_oh = (audience_type[:, None] == audience_types).astype(float)

    X_dgp = np.column_stack(
        [
            obj_oh,
            vert_oh,
            aud_oh,
            budget / 100_000,
            exposure_rate,
            ctr,
            avg_treated_outcome / 5.0,
        ]
    )

    betas = np.array(
        [0.30, -0.10, 0.20, 0.40, -0.20, 0.10, 0.50, -0.30, 0.20, 0.60, -0.10, 0.30]
    )

    tau_true = X_dgp @ betas
    y_observed = rng.normal(tau_true, scale=0.1)
    last_click = np.clip(
        0.2 + 0.65 * tau_true + rng.normal(0.0, scale=0.25, size=n_campaigns),
        0.0,
        None,
    )

    return pd.DataFrame(
        {
            "campaign_id": [f"c_{i:04d}" for i in range(n_campaigns)],
            "objective": objective,
            "vertical": vertical,
            "audience_type": audience_type,
            "budget": budget,
            "exposure_rate": exposure_rate,
            "ctr": ctr,
            "avg_treated_outcome": avg_treated_outcome,
            "last_click_conversions_per_dollar": last_click,
            "measured_incrementality_per_dollar": y_observed,
        }
    )


def test_synthetic_corpus_schema():
    """Generated corpus has the documented columns, dtypes, and value ranges."""
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
    """The same seed yields an identical corpus."""
    df1 = generate_synthetic_rct_corpus(n_campaigns=20, seed=7)
    df2 = generate_synthetic_rct_corpus(n_campaigns=20, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_corpus_different_seeds():
    """Different seeds yield different incrementality labels."""
    df1 = generate_synthetic_rct_corpus(n_campaigns=20, seed=1)
    df2 = generate_synthetic_rct_corpus(n_campaigns=20, seed=2)
    assert not df1["measured_incrementality_per_dollar"].equals(
        df2["measured_incrementality_per_dollar"]
    )


@pytest.fixture(scope="module")
def small_corpus():
    """A 50-row corpus split into the feature matrix X and target y."""
    df = generate_synthetic_rct_corpus(n_campaigns=50, seed=42)
    X = df.drop(columns=["campaign_id", "measured_incrementality_per_dollar"])
    y = df["measured_incrementality_per_dollar"]
    return X, y


@pytest.fixture
def default_model():
    """Fresh PIEModel per test, since tests mutate it via build_model."""
    return PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
    )


def test_build_model(default_model, small_corpus):
    """build_model wires up the BART graph, sigma, and coords."""
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
    assert list(model.coords["feature"]) == PRE + POST


def test_dtype_categorical_detection(default_model, small_corpus):
    """Object-dtype columns get encoders; numeric columns do not."""
    X, y = small_corpus
    default_model.build_model(X, y)

    for col in ("objective", "vertical", "audience_type"):
        assert col in default_model._encoders, f"Expected encoder for {col}"

    for col in ("budget", "exposure_rate", "ctr"):
        assert col not in default_model._encoders, f"Unexpected encoder for {col}"


def test_string_dtype_feature_is_encoded(default_model, small_corpus):
    """Pandas StringDtype columns are detected as categorical and encoded."""
    X, y = small_corpus
    X_string = X.astype({"objective": "string"})
    assert isinstance(X_string["objective"].dtype, pd.StringDtype)

    default_model.build_model(X_string, y)

    assert "objective" in default_model._encoders, (
        "string-dtype column should have been label-encoded"
    )
    default_model._data_setter(X_string, y=None)


def test_model_config_override(small_corpus):
    """model_config overrides reach the built model."""
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
    """A declared feature absent from X raises ValueError."""
    X, y = small_corpus
    bad_model = PIEModel(
        pre_determined_features=["objective", "MISSING_FEATURE"],
        post_determined_features=POST,
    )
    with pytest.raises(ValueError, match="MISSING_FEATURE"):
        bad_model.build_model(X, y)


def test_negative_incrementality_is_allowed(default_model, small_corpus):
    """Negative incrementality values are accepted."""
    X, y = small_corpus
    y_with_negative_values = y - y.mean()
    default_model.build_model(X, y_with_negative_values)


def test_pymc_bart_missing_raises(small_corpus, monkeypatch):
    """build_model raises ImportError when pymc-bart is unavailable."""
    import pymc_marketing.pie.model as pie_module

    monkeypatch.setattr(pie_module, "pmb", None)

    model = PIEModel(pre_determined_features=PRE, post_determined_features=POST)
    X, y = small_corpus
    with pytest.raises(ImportError, match="pymc-marketing\\[pie\\]"):
        model.build_model(X, y)


def test_build_model_length_mismatch_raises(default_model, small_corpus):
    """Mismatched X and y lengths raise ValueError."""
    X, y = small_corpus
    with pytest.raises(ValueError, match="same length"):
        default_model.build_model(X, y.iloc[:-1])


def test_build_model_non_finite_y_raises(default_model, small_corpus):
    """Non-finite y raises ValueError."""
    X, y = small_corpus
    y_bad = y.copy()
    y_bad.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        default_model.build_model(X, y_bad)


def test_build_model_invalid_categorical_split_raises(small_corpus):
    """An invalid categorical_split raises ValueError."""
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
            "categorical_split": "not_a_valid_choice",
        },
    )
    with pytest.raises(ValueError, match="categorical_split"):
        model.build_model(X, y)


def test_build_model_continuous_split_rules(small_corpus):
    """categorical_split='continuous' builds without error."""
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
            "categorical_split": "continuous",
        },
    )
    model.build_model(X, y)
    assert model.model_config["categorical_split"] == "continuous"


def test_partial_bart_override_raises(small_corpus):
    """A partial bart override raises a clear ValueError, not a bare KeyError."""
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={"bart": {"m": 50}},
    )
    with pytest.raises(ValueError, match="missing required keys"):
        model.build_model(X, y)


def test_bart_response_override(small_corpus):
    """A 'linear' BART response builds; an invalid response raises ValueError."""
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0, "response": "linear"},
            "sigma": Prior("HalfNormal", sigma=1.0),
        },
    )
    model.build_model(X, y)

    bad = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={
            "bart": {"m": 10, "alpha": 0.95, "beta": 2.0, "response": "nonsense"},
            "sigma": Prior("HalfNormal", sigma=1.0),
        },
    )
    with pytest.raises(ValueError, match="response"):
        bad.build_model(X, y)


def test_extra_columns_in_X_are_ignored(small_corpus):
    """Columns outside the pre/post features are dropped, not trained on."""
    X, y = small_corpus
    model = PIEModel(pre_determined_features=PRE, post_determined_features=POST)
    X_extra = X.assign(campaign_note="ignore-me")

    model.build_model(X_extra, y)

    assert model._feature_columns == PRE + POST
    assert list(model.model.coords["feature"]) == PRE + POST
    assert "campaign_note" not in model._encoders


def test_output_var_conflict_in_X_raises(small_corpus):
    """fit rejects an X containing a column named like the target variable."""
    X, y = small_corpus
    model = PIEModel(pre_determined_features=PRE, post_determined_features=POST)
    X_conflict = X.assign(y=0.0)
    with pytest.raises(ValueError, match="conflicts with the target variable"):
        model.fit(X_conflict, y)


def test_unseen_categorical_raises(default_model, small_corpus):
    """An unseen category at predict time raises ValueError."""
    X, y = small_corpus
    default_model.build_model(X, y)

    X_bad = X.copy()
    X_bad.loc[X_bad.index[0], "objective"] = "UNSEEN_LEVEL"

    with pytest.raises(ValueError, match="objective"):
        default_model._data_setter(X_bad)


def test_data_setter_dummy_y_when_none(default_model, small_corpus):
    """_data_setter installs zeros for the observed y in predict mode."""
    X, y = small_corpus
    default_model.build_model(X, y)
    default_model._data_setter(X, y=None)
    y_obs_value = default_model.model["y_obs"].get_value()
    np.testing.assert_array_equal(y_obs_value, np.zeros(len(X)))


def test_data_setter_requires_training_feature_schema(default_model, small_corpus):
    """_data_setter requires the training features and ignores extra columns."""
    X, y = small_corpus
    default_model.build_model(X, y)

    with pytest.raises(ValueError, match=r"missing.*budget"):
        default_model._data_setter(X.drop(columns=["budget"]))

    default_model._data_setter(X.assign(extra_feature=1.0), y=None)


def test_data_setter_reorders_columns_to_training_order(default_model, small_corpus):
    """_data_setter tolerates columns supplied in a different order."""
    X, y = small_corpus
    default_model.build_model(X, y)

    default_model._data_setter(X[list(reversed(X.columns))], y=None)


def test_data_setter_with_y_rescales(default_model, small_corpus):
    """A y passed to _data_setter is divided by _target_scale."""
    X, y = small_corpus
    default_model.build_model(X, y)

    X_new = X.iloc[:5].copy()
    y_new = y.iloc[:5]
    default_model._data_setter(X_new, y=y_new)

    y_obs_value = default_model.model["y_obs"].get_value()
    expected = y_new.to_numpy() / default_model._target_scale
    np.testing.assert_allclose(y_obs_value, expected)


def test_data_setter_y_length_mismatch_raises(default_model, small_corpus):
    """Mismatched X and y lengths raise ValueError in _data_setter."""
    X, y = small_corpus
    default_model.build_model(X, y)
    with pytest.raises(ValueError, match="same length"):
        default_model._data_setter(X.iloc[:5], y=y.iloc[:4])


def test_data_setter_non_finite_y_raises(default_model, small_corpus):
    """Non-finite y raises ValueError in _data_setter."""
    X, y = small_corpus
    default_model.build_model(X, y)
    y_bad = y.iloc[:5].copy()
    y_bad.iloc[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        default_model._data_setter(X.iloc[:5], y=y_bad)


def test_sample_posterior_predictive_unscaled(small_corpus):
    """Posterior predictive draws are returned in the original target scale."""
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
    assert pred_max > 100, f"Predictions look scaled: pred_max={pred_max:.4f}"
    assert stored_max > 100, (
        f"Stored idata predictions look scaled: stored_max={stored_max:.4f}"
    )


def test_sample_posterior_predictive_supports_new_obs_count(small_corpus):
    """Prediction works for an observation count different from training."""
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


def test_save_load_roundtrip(small_corpus, tmp_path):
    """A saved model reloads with its config and predicts."""
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


def test_predict_returns_unscaled_means(small_corpus):
    """predict returns means in the original target scale."""
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
    assert abs(preds).max() > 100, (
        f"predict() looks scaled: max(|preds|)={abs(preds).max():.4f}"
    )


def test_predict_posterior_returns_unscaled_draws(small_corpus):
    """predict_posterior returns draws in the original target scale."""
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

    draws = model.predict_posterior(X, extend_idata=False)

    assert draws.sizes["obs"] == len(X)
    assert float(abs(draws).max()) > 100, (
        f"predict_posterior looks scaled: max={float(abs(draws).max()):.4f}"
    )


def test_partial_model_config_override(small_corpus):
    """Omitted top-level model_config keys fall back to defaults."""
    X, y = small_corpus
    model = PIEModel(
        pre_determined_features=PRE,
        post_determined_features=POST,
        model_config={"bart": {"m": 25, "alpha": 0.95, "beta": 2.0}},
    )
    assert model.model_config["bart"]["m"] == 25

    assert isinstance(model.model_config["sigma"], Prior)
    assert model.model_config["sigma"].distribution == "HalfNormal"
    assert model.model_config["sigma"].parameters["sigma"] == 1.0

    assert model.model_config["categorical_split"] == "onehot"

    model.build_model(X, y)


@pytest.mark.slow
def test_fit_recovers_known_effects():
    """Predictions correlate with held-out incrementality and match its scale."""
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

    mean_abs_pred = float(np.abs(pred_mean).mean())
    assert 0.05 < mean_abs_pred < 10.0, (
        f"Predictions off the expected scale: mean|pred|={mean_abs_pred:.4f}"
    )
