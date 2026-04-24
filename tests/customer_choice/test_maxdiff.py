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
"""Unit and parametric-recovery tests for MaxDiffMixedLogit."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pymc_marketing.customer_choice.maxdiff import (
    MaxDiffMixedLogit,
    prepare_maxdiff_data,
)
from pymc_marketing.customer_choice.synthetic_data import generate_maxdiff_data


def _small_maxdiff_df(
    n_respondents: int = 6,
    n_items: int = 5,
    n_tasks_per_resp: int = 4,
    subset_size: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Build a small synthetic MaxDiff design for fast tests."""
    task_df, ground_truth = generate_maxdiff_data(
        n_respondents=n_respondents,
        n_items=n_items,
        n_tasks_per_resp=n_tasks_per_resp,
        subset_size=subset_size,
        sigma_respondent=0.3,
        random_seed=seed,
    )
    return task_df, ground_truth["items"], ground_truth


@pytest.fixture
def small_maxdiff():
    return _small_maxdiff_df()


@pytest.fixture
def ragged_maxdiff():
    """Build a tiny task_df with mixed subset sizes (3 and 4)."""
    rng = np.random.default_rng(3)
    items = [f"item_{i}" for i in range(6)]
    rows = []
    for r in range(4):
        for t in range(3):
            k = 3 if t % 2 == 0 else 4
            subset = rng.choice(len(items), size=k, replace=False)
            # Deterministic best / worst within the subset
            best_local = 0
            worst_local = k - 1
            for local_pos, idx in enumerate(subset):
                rows.append(
                    {
                        "respondent_id": f"r{r}",
                        "task_id": t,
                        "item_id": items[idx],
                        "is_best": int(local_pos == best_local),
                        "is_worst": int(local_pos == worst_local),
                    }
                )
    return pd.DataFrame(rows), items


FAST_FIT_KWARGS = {
    "draws": 40,
    "tune": 40,
    "chains": 1,
    "target_accept": 0.8,
    "nuts_sampler": "pymc",
}


# ---------------------------------------------------------------------------
# prepare_maxdiff_data
# ---------------------------------------------------------------------------


class TestPrepareMaxDiffData:
    def test_shapes(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        arrays = prepare_maxdiff_data(task_df, items=items)

        n_tasks_expected = task_df.groupby(
            ["respondent_id", "task_id"], sort=False
        ).ngroups
        assert arrays["n_tasks"] == n_tasks_expected
        assert arrays["n_items"] == len(items)
        assert arrays["item_idx"].shape == (n_tasks_expected, arrays["k_max"])
        assert arrays["mask"].shape == (n_tasks_expected, arrays["k_max"])
        assert arrays["best_pos"].shape == (n_tasks_expected,)
        assert arrays["worst_pos"].shape == (n_tasks_expected,)
        assert arrays["resp_idx"].shape == (n_tasks_expected,)

    def test_best_worst_pos_match_raw_flags(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        arrays = prepare_maxdiff_data(task_df, items=items)
        item_to_idx = arrays["item_to_idx"]

        grouped = task_df.groupby(["respondent_id", "task_id"], sort=False)
        for t, (_, group) in enumerate(grouped):
            best_item = group.loc[group["is_best"] == 1, "item_id"].iloc[0]
            worst_item = group.loc[group["is_worst"] == 1, "item_id"].iloc[0]
            bp = int(arrays["best_pos"][t])
            wp = int(arrays["worst_pos"][t])
            assert arrays["item_idx"][t, bp] == item_to_idx[best_item]
            assert arrays["item_idx"][t, wp] == item_to_idx[worst_item]

    def test_validation_missing_best(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        df = task_df.copy()
        first_group = (df["respondent_id"] == df["respondent_id"].iloc[0]) & (
            df["task_id"] == df["task_id"].iloc[0]
        )
        df.loc[first_group, "is_best"] = 0  # wipe best for one task
        with pytest.raises(ValueError, match="best"):
            prepare_maxdiff_data(df, items=items)

    def test_validation_duplicate_best(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        df = task_df.copy()
        first_group = (df["respondent_id"] == df["respondent_id"].iloc[0]) & (
            df["task_id"] == df["task_id"].iloc[0]
        )
        df.loc[first_group, "is_best"] = 1  # all ones
        with pytest.raises(ValueError, match="best"):
            prepare_maxdiff_data(df, items=items)

    def test_validation_best_equals_worst(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        df = task_df.copy()
        first_group = (df["respondent_id"] == df["respondent_id"].iloc[0]) & (
            df["task_id"] == df["task_id"].iloc[0]
        )
        # Force is_worst to mirror is_best on this task
        df.loc[first_group, "is_worst"] = df.loc[first_group, "is_best"]
        with pytest.raises(ValueError, match="best and worst"):
            prepare_maxdiff_data(df, items=items)

    def test_validation_unknown_item(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        df = task_df.copy()
        df.loc[0, "item_id"] = "not_in_pool"
        with pytest.raises(ValueError, match="not found"):
            prepare_maxdiff_data(df, items=items)

    def test_validation_duplicate_item_in_task(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        df = task_df.copy()
        first_group_idx = df.index[
            (df["respondent_id"] == df["respondent_id"].iloc[0])
            & (df["task_id"] == df["task_id"].iloc[0])
        ]
        # Make the first two shown items identical in that task
        df.loc[first_group_idx[1], "item_id"] = df.loc[first_group_idx[0], "item_id"]
        with pytest.raises(ValueError, match="duplicate"):
            prepare_maxdiff_data(df, items=items)

    def test_validation_missing_columns(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        df = task_df.drop(columns=["is_worst"])
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_maxdiff_data(df, items=items)


# ---------------------------------------------------------------------------
# Build / logp / config
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_build_model_smoke(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.build_model()
        assert model.model is not None

        for coord in ("items", "respondents", "tasks", "positions"):
            assert coord in model.model.coords

        assert list(model.model.coords["items"]) == items

        var_names = [v.name for v in model.model.free_RVs]
        assert "beta_item_" in var_names
        assert "sigma_item" in var_names
        assert "z_item" in var_names

        det_names = [d.name for d in model.model.deterministics]
        for d in ("beta_item", "U", "p_best", "p_worst", "beta_item_r"):
            assert d in det_names

    def test_reference_item_pinned_at_zero(self, small_maxdiff):
        import pymc as pm

        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.build_model()
        with model.model:
            beta_item = pm.draw(model.model["beta_item"], random_seed=0)
        ref_pos = items.index(items[-1])
        assert beta_item[ref_pos] == 0.0

    def test_model_logp_finite(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.build_model()
        logp_fn = model.model.compile_logp()
        init = model.model.initial_point()
        logp_val = float(logp_fn(init))
        assert np.isfinite(logp_val)

    def test_reference_item_custom(self, small_maxdiff):
        import pymc as pm

        task_df, items, _ = small_maxdiff
        custom_ref = items[0]
        model = MaxDiffMixedLogit(
            task_df=task_df, items=items, reference_item=custom_ref
        )
        model.build_model()
        with model.model:
            beta_item = pm.draw(model.model["beta_item"], random_seed=0)
        assert beta_item[0] == 0.0

    def test_reference_item_not_in_pool_raises(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        with pytest.raises(ValueError, match="reference_item"):
            MaxDiffMixedLogit(
                task_df=task_df, items=items, reference_item="totally_made_up"
            )

    def test_random_intercepts_toggle(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items, random_intercepts=False)
        model.build_model()
        var_names = [v.name for v in model.model.free_RVs]
        assert "z_item" not in var_names
        assert "sigma_item" not in var_names
        assert "beta_item_" in var_names

        det_names = [d.name for d in model.model.deterministics]
        assert "beta_item_r" not in det_names

    def test_ragged_subset_sizes(self, ragged_maxdiff):
        task_df, items = ragged_maxdiff
        arrays = prepare_maxdiff_data(task_df, items=items)
        # k_max should be the larger subset size (4)
        assert arrays["k_max"] == 4
        # Shorter tasks should have at least one False mask position
        short_rows = (~arrays["mask"]).any(axis=1)
        assert short_rows.any()

        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.build_model()
        logp_fn = model.model.compile_logp()
        init = model.model.initial_point()
        assert np.isfinite(float(logp_fn(init)))


class TestSerializableConfig:
    def test_model_type(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        assert model._model_type == "MaxDiff Mixed Logit"

    def test_serializable_config_includes_random(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        config = model._serializable_model_config
        assert "beta_item_" in config
        assert "sigma_item" in config
        assert "z_item" in config

    def test_serializable_config_fixed_only(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items, random_intercepts=False)
        config = model._serializable_model_config
        assert "beta_item_" in config
        assert "sigma_item" not in config
        assert "z_item" not in config


# ---------------------------------------------------------------------------
# Prior predictive / save-load / fit-dependent tests
# ---------------------------------------------------------------------------


class TestPriorPredictive:
    def test_prior_predictive(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        prior = model.sample_prior_predictive(samples=10, random_seed=42)
        assert "prior_predictive" in prior.groups()
        assert "best_pick" in prior["prior_predictive"]
        assert "worst_pick" in prior["prior_predictive"]


class TestSaveLoadRoundtrip:
    def test_save_load_roundtrip(self, small_maxdiff, tmp_path):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)

        path = tmp_path / "maxdiff.nc"
        model.save(str(path))

        loaded = MaxDiffMixedLogit.load(str(path))
        assert loaded.items == items
        assert loaded.respondent_id == model.respondent_id
        assert loaded.task_id == model.task_id
        assert loaded.item_col == model.item_col
        assert loaded.best_col == model.best_col
        assert loaded.worst_col == model.worst_col
        assert loaded.random_intercepts == model.random_intercepts
        assert loaded.reference_item == model.reference_item
        assert loaded.non_centered == model.non_centered

        # task_df should have been restored from fit_data
        assert len(loaded.task_df) == len(task_df)
        for col in ("respondent_id", "task_id", "item_id", "is_best", "is_worst"):
            assert col in loaded.task_df.columns


class TestPosteriorPredictive:
    def test_posterior_predictive_shapes(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        idata = model.fit(random_seed=42, **FAST_FIT_KWARGS)
        assert idata is not None

        post_pred = model.sample_posterior_predictive(random_seed=42)
        assert "posterior_predictive" in post_pred.groups()
        pp = post_pred["posterior_predictive"]
        assert "best_pick" in pp
        assert "worst_pick" in pp

        n_tasks = model.arrays["n_tasks"]
        chains = FAST_FIT_KWARGS["chains"]
        draws = FAST_FIT_KWARGS["draws"]
        assert pp["best_pick"].shape == (chains, draws, n_tasks)
        assert pp["worst_pick"].shape == (chains, draws, n_tasks)


class TestPredictChoices:
    def test_counterfactual_predict_choices(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)

        preds = model.predict_choices(task_df, random_seed=123)
        assert "best_pick" in preds
        assert "worst_pick" in preds
        assert "p_best" in preds
        assert "p_worst" in preds

        chains = FAST_FIT_KWARGS["chains"]
        draws = FAST_FIT_KWARGS["draws"]
        n_tasks = model.arrays["n_tasks"]
        k_max = model.arrays["k_max"]
        assert preds["best_pick"].shape == (chains, draws, n_tasks)
        assert preds["worst_pick"].shape == (chains, draws, n_tasks)
        assert preds["p_best"].shape == (chains, draws, n_tasks, k_max)
        assert preds["p_worst"].shape == (chains, draws, n_tasks, k_max)

        best_vals = preds["best_pick"].values
        worst_vals = preds["worst_pick"].values
        # Best and worst must never coincide for the *same draw* (generative
        # worst sampling excludes the sampled best).
        assert not np.any(best_vals == worst_vals)

        # Sampled positions must be real (not padding).
        mask = model.arrays["mask"]
        for t in range(n_tasks):
            allowed = np.where(mask[t])[0]
            assert np.isin(best_vals[..., t], allowed).all()
            assert np.isin(worst_vals[..., t], allowed).all()


# ---------------------------------------------------------------------------
# Parametric recovery (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestParametricRecovery:
    def test_rank_correlation_recovered(self):
        """Posterior-mean item utilities should rank-correlate with truth."""
        task_df, ground_truth = generate_maxdiff_data(
            n_respondents=150,
            n_items=10,
            n_tasks_per_resp=12,
            subset_size=4,
            sigma_respondent=0.4,
            random_seed=7,
        )
        items = ground_truth["items"]
        true_utilities = ground_truth["utilities"]

        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.fit(
            random_seed=42,
            draws=500,
            tune=500,
            chains=2,
            target_accept=0.9,
            nuts_sampler="pymc",
        )

        post_mean = (
            model.idata["posterior"]["beta_item"].mean(dim=("chain", "draw")).values
        )
        rho, _ = spearmanr(post_mean, true_utilities)
        assert rho >= 0.85, f"Spearman rank-correlation too low: {rho}"
