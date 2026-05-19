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

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import spearmanr

from pymc_marketing.customer_choice.maxdiff import (
    MaxDiffMixedLogit,
    prepare_maxdiff_data,
)
from pymc_marketing.customer_choice.synthetic_data import (
    generate_maxdiff_conjoint_data,
    generate_maxdiff_data,
)


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

    def test_duplicate_items_raises(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        dup_items = [*items, items[0]]
        with pytest.raises(ValueError, match="duplicates"):
            prepare_maxdiff_data(task_df, items=dup_items)


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

    def test_init_duplicate_items_raises(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        dup_items = [*items, items[0]]
        with pytest.raises(ValueError, match="duplicates"):
            MaxDiffMixedLogit(task_df=task_df, items=dup_items)

    def test_reference_item_pinned_at_zero(self, small_maxdiff):
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

    def test_default_config_intercept_mode_no_stale_keys(self, small_maxdiff):
        """Item-intercept default config must not contain part-worths keys."""
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        cfg = model.default_model_config
        assert "beta_item_" in cfg
        assert "beta_feat" not in cfg
        assert "sigma_feat" not in cfg

    def test_default_config_partworths_mode_no_stale_keys(self, partworths_fixture):
        """Part-worths default config must not contain intercept-mode keys."""
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
        )
        cfg = model.default_model_config
        assert "beta_feat" in cfg
        assert "beta_item_" not in cfg
        assert "sigma_item" not in cfg
        assert "z_item" not in cfg


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

    def test_task_id_dtype_preserved(self, small_maxdiff, tmp_path):
        task_df, items, _ = small_maxdiff
        original_dtype = task_df["task_id"].dtype
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        path = tmp_path / "maxdiff_dtype.nc"
        model.save(str(path))
        loaded = MaxDiffMixedLogit.load(str(path))
        assert loaded.task_df["task_id"].dtype == original_dtype


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

    def test_shape_mismatch_k_max_raises(self):
        # Train on subset_size=3 (k_max=3), then predict with subset_size=5 (k_max=5)
        task_df_train, items, _ = _small_maxdiff_df(
            n_respondents=4, n_items=5, n_tasks_per_resp=3, subset_size=3
        )
        model = MaxDiffMixedLogit(task_df=task_df_train, items=items)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        assert model.arrays["k_max"] == 3

        # Build a task_df where every task shows all 5 items → k_max=5
        rng = np.random.default_rng(99)
        rows = []
        for r in range(2):
            for t in range(2):
                perm = rng.permutation(len(items))
                for local_pos, idx in enumerate(perm):
                    rows.append(
                        {
                            "respondent_id": f"r{r}",
                            "task_id": t,
                            "item_id": items[idx],
                            "is_best": int(local_pos == 0),
                            "is_worst": int(local_pos == len(items) - 1),
                        }
                    )
        big_task_df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="k_max"):
            model.sample_posterior_predictive(task_df=big_task_df)

    def test_shape_mismatch_n_tasks_raises(self):
        # Train on 3 tasks per respondent, then predict with 2 tasks per respondent
        task_df_train, items, _ = _small_maxdiff_df(
            n_respondents=4, n_items=5, n_tasks_per_resp=3, subset_size=3
        )
        model = MaxDiffMixedLogit(task_df=task_df_train, items=items)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)

        task_df_fewer, _, _ = _small_maxdiff_df(
            n_respondents=4, n_items=5, n_tasks_per_resp=2, subset_size=3, seed=1
        )
        assert (
            task_df_fewer.groupby(["respondent_id", "task_id"]).ngroups
            != model.arrays["n_tasks"]
        )
        with pytest.raises(ValueError, match="tasks"):
            model.sample_posterior_predictive(task_df=task_df_fewer)


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


# ---------------------------------------------------------------------------
# v0.2 additions: single-item-task guard, batched predict, new respondents
# ---------------------------------------------------------------------------


class TestSingleItemTaskValidation:
    def test_validation_single_item_task(self):
        """Tasks with fewer than 2 items must be rejected up front."""
        task_df = pd.DataFrame(
            [
                # r0 task 0 is a *valid* 2-item task
                {
                    "respondent_id": "r0",
                    "task_id": 0,
                    "item_id": "a",
                    "is_best": 1,
                    "is_worst": 0,
                },
                {
                    "respondent_id": "r0",
                    "task_id": 0,
                    "item_id": "b",
                    "is_best": 0,
                    "is_worst": 1,
                },
                # r0 task 1 has only one item — undefined for best-worst scaling
                {
                    "respondent_id": "r0",
                    "task_id": 1,
                    "item_id": "a",
                    "is_best": 1,
                    "is_worst": 1,
                },
            ]
        )
        with pytest.raises(ValueError, match="at least 2 items"):
            prepare_maxdiff_data(task_df, items=["a", "b"])


class TestPredictChoicesBatched:
    def test_batched_equivalence(self, small_maxdiff):
        """draw_batch_size must not change the output for a fixed seed."""
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)

        ds_full = model.predict_choices(task_df, random_seed=7)
        ds_batched = model.predict_choices(task_df, random_seed=7, draw_batch_size=11)
        np.testing.assert_array_equal(
            ds_full["best_pick"].values, ds_batched["best_pick"].values
        )
        np.testing.assert_array_equal(
            ds_full["worst_pick"].values, ds_batched["worst_pick"].values
        )
        np.testing.assert_allclose(
            ds_full["p_best"].values, ds_batched["p_best"].values
        )
        np.testing.assert_allclose(
            ds_full["p_worst"].values, ds_batched["p_worst"].values
        )


class TestPredictChoicesNewRespondents:
    @pytest.fixture
    def fitted_model(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        return task_df, items, model

    def test_unknown_respondent_error_default(self, fitted_model):
        task_df, _items, model = fitted_model
        new_df = task_df.copy()
        new_df["respondent_id"] = new_df["respondent_id"].astype(str) + "_new"
        with pytest.raises(ValueError, match="new_respondents='population'"):
            model.predict_choices(new_df, random_seed=0)

    def test_unknown_respondent_population(self, fitted_model):
        task_df, _items, model = fitted_model
        new_df = task_df.copy()
        new_df["respondent_id"] = new_df["respondent_id"].astype(str) + "_new"
        ds = model.predict_choices(new_df, random_seed=0, new_respondents="population")
        n_tasks = len(new_df.groupby(["respondent_id", "task_id"]))
        chains = FAST_FIT_KWARGS["chains"]
        draws = FAST_FIT_KWARGS["draws"]
        assert ds["best_pick"].shape == (chains, draws, n_tasks)
        # Each row of p_best must still sum to 1 across positions.
        np.testing.assert_allclose(
            ds["p_best"].sum(dim="positions").values, 1.0, atol=1e-10
        )

    def test_population_ignored_when_no_random_intercepts(self, small_maxdiff):
        """With random_intercepts=False, unknown respondents are trivially ok."""
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items, random_intercepts=False)
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        new_df = task_df.copy()
        new_df["respondent_id"] = new_df["respondent_id"].astype(str) + "_new"
        # No error, and "population" is a no-op.
        ds = model.predict_choices(new_df, random_seed=0, new_respondents="population")
        assert "best_pick" in ds


# ---------------------------------------------------------------------------
# Part-worths (conjoint-style) MaxDiff
# ---------------------------------------------------------------------------


@pytest.fixture
def partworths_fixture():
    """A tiny part-worths dataset for fast build/logp/fit tests."""
    task_df, attrs, gt = generate_maxdiff_conjoint_data(
        n_respondents=6,
        n_items=6,
        n_tasks_per_resp=4,
        subset_size=3,
        sigma_respondent=0.3,
        random_seed=0,
    )
    return task_df, attrs, gt


class TestPartWorthsBuild:
    def test_build_model_smoke(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
        )
        model.build_model()
        rv_names = [v.name for v in model.model.unobserved_RVs]
        assert "beta_feat" in rv_names
        assert "U_item_pop" in rv_names
        # No item intercepts in part-worths mode.
        assert "beta_item" not in rv_names
        assert "features" in model.coords
        # random_features absent when random_attributes is empty.
        assert "random_features" not in model.coords

    def test_random_attributes_adds_per_respondent(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price"],
        )
        model.build_model()
        rv_names = [v.name for v in model.model.unobserved_RVs]
        assert {"beta_feat", "sigma_feat", "z_feat", "U_item_r"}.issubset(rv_names)
        assert model.coords["random_features"] == ["price"]

    def test_logp_finite(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price"],
        )
        model.build_model()
        logp = model.model.compile_logp()(model.model.initial_point())
        assert np.isfinite(float(logp))

    def test_item_attributes_missing_row_raises(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        # Drop the last item from attrs but keep it in items.
        bad_attrs = attrs.iloc[:-1]
        with pytest.raises(ValueError, match="missing rows"):
            MaxDiffMixedLogit(
                task_df=task_df,
                items=gt["items"],
                item_attributes=bad_attrs,
                utility_formula="~ 0 + C(brand) + price + quality",
            )

    def test_formula_both_or_neither(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        with pytest.raises(ValueError, match="both"):
            MaxDiffMixedLogit(
                task_df=task_df,
                items=gt["items"],
                item_attributes=attrs,
                utility_formula=None,
            )
        with pytest.raises(ValueError, match="both"):
            MaxDiffMixedLogit(
                task_df=task_df,
                items=gt["items"],
                item_attributes=None,
                utility_formula="~ 0 + C(brand) + price + quality",
            )

    def test_random_attributes_requires_partworths(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        with pytest.raises(ValueError, match="part-worths"):
            MaxDiffMixedLogit(
                task_df=task_df,
                items=items,
                random_attributes=["price"],
            )

    def test_random_attributes_unknown_feature(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        with pytest.raises(ValueError, match="not found in expanded feature"):
            MaxDiffMixedLogit(
                task_df=task_df,
                items=gt["items"],
                item_attributes=attrs,
                utility_formula="~ 0 + C(brand) + price + quality",
                random_attributes=["not_a_feature"],
            )

    def test_transform_attributes_round_trips_existing(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
        )
        # Re-applying the formula to the original frame must reproduce model.X.
        X_round = model.transform_attributes(attrs.loc[gt["items"]])
        np.testing.assert_allclose(X_round, model.X)

    def test_transform_attributes_new_row(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
        )
        new_row = pd.DataFrame(
            [{"brand": "B", "price": 0.42, "quality": 1.5}],
            index=pd.Index(["item_NEW"], name="item_id"),
        )
        X_new = model.transform_attributes(new_row)
        assert X_new.shape == (1, len(model.feature_names))
        # Reproduces what we'd get by passing the brand level explicitly.
        assert "C(brand)[B]" in model.feature_names
        b_col = model.feature_names.index("C(brand)[B]")
        price_col = model.feature_names.index("price")
        quality_col = model.feature_names.index("quality")
        assert X_new[0, b_col] == 1.0
        np.testing.assert_allclose(X_new[0, price_col], 0.42)
        np.testing.assert_allclose(X_new[0, quality_col], 1.5)

    def test_transform_attributes_requires_partworths(self, small_maxdiff):
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        with pytest.raises(RuntimeError, match="part-worths mode"):
            model.transform_attributes(pd.DataFrame({"brand": ["A"]}))


class TestPartWorthsPosterior:
    def test_posterior_predictive_shapes(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price"],
        )
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        pp = model.sample_posterior_predictive(random_seed=42)
        n_tasks = model.arrays["n_tasks"]
        assert pp["posterior_predictive"]["best_pick"].shape == (
            FAST_FIT_KWARGS["chains"],
            FAST_FIT_KWARGS["draws"],
            n_tasks,
        )

    def test_predict_choices_shapes(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price"],
        )
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        ds = model.predict_choices(task_df, random_seed=0)
        n_tasks = model.arrays["n_tasks"]
        k_max = model.arrays["k_max"]
        assert ds["best_pick"].shape == (
            FAST_FIT_KWARGS["chains"],
            FAST_FIT_KWARGS["draws"],
            n_tasks,
        )
        assert ds["p_best"].shape == (
            FAST_FIT_KWARGS["chains"],
            FAST_FIT_KWARGS["draws"],
            n_tasks,
            k_max,
        )
        # Each task's p_best must sum to 1 across positions.
        np.testing.assert_allclose(
            ds["p_best"].sum(dim="positions").values, 1.0, atol=1e-10
        )

    def test_predict_choices_new_respondent_population(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price"],
        )
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        new_df = task_df.copy()
        new_df["respondent_id"] = new_df["respondent_id"].astype(str) + "_new"
        with pytest.raises(ValueError):
            model.predict_choices(new_df, random_seed=0)
        ds = model.predict_choices(new_df, random_seed=0, new_respondents="population")
        assert "best_pick" in ds

    def test_new_respondent_draws_centered_like_population(self, partworths_fixture):
        """New-respondent utility draws must be on the same scale as U_item_pop."""
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price"],
        )
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        posterior = model.idata["posterior"]
        rng = np.random.default_rng(0)
        draws = model._draw_new_respondent_utilities(posterior, n_new=50, rng=rng)
        # Mean over new respondents and items should be close to mean of U_item_pop
        u_pop_mean = float(posterior["U_item_pop"].mean().values)
        draws_mean = float(draws.mean())
        assert abs(draws_mean - u_pop_mean) < 0.5


class TestPartWorthsSaveLoad:
    def test_save_load_roundtrip(self, partworths_fixture, tmp_path):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price"],
        )
        model.fit(random_seed=42, **FAST_FIT_KWARGS)
        path = tmp_path / "maxdiff_partworths.nc"
        model.save(str(path))
        reloaded = MaxDiffMixedLogit.load(str(path))
        assert reloaded.utility_formula == model.utility_formula
        assert reloaded.random_attributes == model.random_attributes
        assert reloaded.feature_names == model.feature_names
        # item_attributes rehydrated with same index and columns
        pd.testing.assert_frame_equal(
            reloaded.item_attributes.loc[model.items],
            model.item_attributes.loc[model.items],
            check_dtype=False,
        )


@pytest.mark.slow
class TestPartWorthsRecovery:
    def test_partworths_rank_correlation_recovered(self):
        """Population part-worths should be rank-recovered from synthetic data."""
        task_df, attrs, gt = generate_maxdiff_conjoint_data(
            n_respondents=150,
            n_items=10,
            n_tasks_per_resp=12,
            subset_size=4,
            sigma_respondent=0.3,
            random_attributes=["price", "quality"],
            random_seed=7,
        )
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            random_attributes=["price", "quality"],
        )
        model.fit(
            random_seed=42,
            draws=500,
            tune=500,
            chains=2,
            target_accept=0.9,
            nuts_sampler="pymc",
        )
        # Compare item utilities, not raw coefficients
        post_U_mean = (
            model.idata["posterior"]["U_item_pop"].mean(dim=("chain", "draw")).values
        )
        true_U = gt["X"] @ gt["betas"]  # True item utilities
        # Center both (identification is up to shift)
        true_U_centered = true_U - true_U.mean()
        post_U_centered = post_U_mean - post_U_mean.mean()

        rho, _ = spearmanr(post_U_centered, true_U_centered)
        assert rho >= 0.85, f"Spearman rank-correlation too low: {rho}"


# ---------------------------------------------------------------------------
# apply_intervention and score_new_items
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_intercept_model(small_maxdiff):
    """Fitted item-intercept model for intervention tests."""
    task_df, items, _ = small_maxdiff
    model = MaxDiffMixedLogit(task_df=task_df, items=items)
    model.fit(random_seed=42, **FAST_FIT_KWARGS)
    return task_df, items, model


@pytest.fixture
def fitted_partworths_model(partworths_fixture):
    """Fitted part-worths model for intervention tests."""
    task_df, attrs, gt = partworths_fixture
    model = MaxDiffMixedLogit(
        task_df=task_df,
        items=gt["items"],
        item_attributes=attrs,
        utility_formula="~ 0 + C(brand) + price + quality",
    )
    model.fit(random_seed=42, **FAST_FIT_KWARGS)
    return task_df, attrs, gt, model


class TestApplyIntervention:
    def test_returns_xr_dataset(self, fitted_intercept_model):
        """apply_intervention must return an xr.Dataset from predict_choices."""
        task_df, _items, model = fitted_intercept_model
        result = model.apply_intervention(task_df, random_seed=0)
        assert isinstance(result, xr.Dataset)
        assert "best_pick" in result
        assert "worst_pick" in result

    def test_stored_as_intervention_idata(self, fitted_intercept_model):
        task_df, _items, model = fitted_intercept_model
        result = model.apply_intervention(task_df, random_seed=0)
        assert isinstance(result, xr.Dataset)
        assert isinstance(model.intervention_idata, az.InferenceData)
        assert "posterior_predictive" in model.intervention_idata.groups()

    def test_auto_generates_dummy_flags(self, fitted_intercept_model):
        """Calling without is_best/is_worst columns must succeed and warn."""
        task_df, _items, model = fitted_intercept_model
        df_no_flags = task_df.drop(columns=["is_best", "is_worst"])
        with pytest.warns(UserWarning, match="Dummy flags"):
            result = model.apply_intervention(df_no_flags, random_seed=0)
        assert "best_pick" in result

    def test_partial_flags_raises(self, fitted_intercept_model):
        """Having exactly one flag column should raise a clear ValueError."""
        task_df, _items, model = fitted_intercept_model
        df_one_flag = task_df.drop(columns=["is_worst"])
        with pytest.raises(ValueError, match="absent"):
            model.apply_intervention(df_one_flag, random_seed=0)

    def test_raises_before_fit(self, small_maxdiff):
        """apply_intervention without fit must raise RuntimeError."""
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        with pytest.raises(RuntimeError, match="fitted model"):
            model.apply_intervention(task_df, random_seed=0)

    def test_output_shape_matches_predict_choices(self, fitted_intercept_model):
        """apply_intervention output shape must match predict_choices directly."""
        task_df, _items, model = fitted_intercept_model
        ds_direct = model.predict_choices(task_df, random_seed=99)
        ds_ai = model.apply_intervention(task_df, random_seed=99)
        assert ds_ai["best_pick"].shape == ds_direct["best_pick"].shape
        np.testing.assert_array_equal(
            ds_ai["best_pick"].values, ds_direct["best_pick"].values
        )


class TestScoreNewItems:
    def test_returns_xr_dataset_with_extended_items(self, fitted_partworths_model):
        """score_new_items must include training items + new items in coord."""
        _task_df, _attrs, gt, model = fitted_partworths_model
        new_item = pd.DataFrame(
            [{"brand": "B", "price": 0.10, "quality": 0.0}],
            index=pd.Index(["item_NEW"], name="item_id"),
        )
        result = model.score_new_items(new_item)
        assert isinstance(result, xr.Dataset)
        assert "share_of_preference" in result
        assert "u_item" in result
        all_items = list(result.coords["items"].values)
        assert "item_NEW" in all_items
        # All training items must come first.
        assert all_items[: len(gt["items"])] == list(gt["items"])

    def test_shares_sum_to_one(self, fitted_partworths_model):
        """Per-draw shares over the extended pool must sum to 1."""
        _task_df, _attrs, _gt, model = fitted_partworths_model
        new_item = pd.DataFrame(
            [{"brand": "A", "price": 0.5, "quality": 0.5}],
            index=pd.Index(["item_NEW"], name="item_id"),
        )
        result = model.score_new_items(new_item)
        np.testing.assert_allclose(
            result["share_of_preference"].sum(dim="items").values,
            1.0,
            atol=1e-10,
        )

    def test_stored_as_intervention_idata(self, fitted_partworths_model):
        _task_df, _attrs, _gt, model = fitted_partworths_model
        new_item = pd.DataFrame(
            [{"brand": "C", "price": 0.2, "quality": -0.5}],
            index=pd.Index(["item_NEW"], name="item_id"),
        )
        result = model.score_new_items(new_item)
        assert isinstance(result, xr.Dataset)
        assert isinstance(model.intervention_idata, az.InferenceData)
        assert "posterior_predictive" in model.intervention_idata.groups()

    def test_requires_partworths_mode(self, fitted_intercept_model):
        _task_df, _items, model = fitted_intercept_model
        new_item = pd.DataFrame(
            [{"brand": "A", "price": 0.5}],
            index=pd.Index(["item_NEW"], name="item_id"),
        )
        with pytest.raises(RuntimeError, match="part-worths mode"):
            model.score_new_items(new_item)

    def test_raises_on_name_overlap(self, fitted_partworths_model):
        _task_df, _attrs, gt, model = fitted_partworths_model
        # Use an existing training item name.
        overlap_item = pd.DataFrame(
            [{"brand": "A", "price": 0.5, "quality": 0.0}],
            index=pd.Index([gt["items"][0]], name="item_id"),
        )
        with pytest.raises(ValueError, match="training pool"):
            model.score_new_items(overlap_item)

    def test_raises_before_fit(self, partworths_fixture):
        task_df, attrs, gt = partworths_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=gt["items"],
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
        )
        new_item = pd.DataFrame(
            [{"brand": "B", "price": 0.1, "quality": 0.0}],
            index=pd.Index(["item_NEW"], name="item_id"),
        )
        with pytest.raises(RuntimeError, match="fitted model"):
            model.score_new_items(new_item)


# ---------------------------------------------------------------------------
# Synthetic data generators — validation paths
# ---------------------------------------------------------------------------


class TestGenerateMaxdiffData:
    """Branch coverage for generate_maxdiff_data validation."""

    def test_subset_size_too_small_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_maxdiff_data(n_items=5, subset_size=1)

    def test_subset_size_exceeds_items_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            generate_maxdiff_data(n_items=4, subset_size=5)

    def test_items_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            generate_maxdiff_data(
                n_items=4,
                subset_size=2,
                items=["a", "b"],  # only 2, but n_items=4
            )

    def test_custom_items_and_utilities(self):
        """Custom items list and true_utilities are threaded through correctly."""
        items = ["alpha", "beta", "gamma", "delta"]
        true_u = np.array([1.0, 0.5, -0.5, 0.0])
        task_df, gt = generate_maxdiff_data(
            n_respondents=3,
            n_items=4,
            subset_size=2,
            items=items,
            true_utilities=true_u,
            random_seed=0,
        )
        assert gt["items"] == items
        # Reference (last) item pinned to 0.
        assert gt["utilities"][-1] == pytest.approx(0.0)
        assert set(task_df["item_id"].unique()) <= set(items)

    def test_item_correlation_diagonal_returns_eye(self):
        """Without item_correlation, ground_truth['item_correlation'] is identity."""
        _task_df, gt = generate_maxdiff_data(
            n_respondents=4, n_items=3, subset_size=2, random_seed=0
        )
        np.testing.assert_array_equal(gt["item_correlation"], np.eye(3))

    def test_item_correlation_stored_in_ground_truth(self):
        """Supplied item_correlation must be echoed back in ground_truth."""
        corr = np.array([[1.0, 0.8, -0.4], [0.8, 1.0, -0.3], [-0.4, -0.3, 1.0]])
        _task_df, gt = generate_maxdiff_data(
            n_respondents=5,
            n_items=3,
            subset_size=2,
            item_correlation=corr,
            random_seed=1,
        )
        np.testing.assert_allclose(gt["item_correlation"], corr)

    def test_item_correlation_produces_correlated_utilities(self):
        """Strong off-diagonal correlations should be visible in respondent utilities."""
        # 3 items: item_0 and item_1 highly correlated, item_2 independent.
        corr = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]])
        _task_df, gt = generate_maxdiff_data(
            n_respondents=2000,
            n_items=3,
            subset_size=2,
            sigma_respondent=1.0,
            item_correlation=corr,
            random_seed=42,
        )
        devs = gt["respondent_utilities"] - gt["utilities"][None, :]
        empirical_corr = np.corrcoef(devs.T)
        # item_0 ↔ item_1 should be strongly positive
        assert empirical_corr[0, 1] > 0.85
        # item_0 ↔ item_2 should be near zero
        assert abs(empirical_corr[0, 2]) < 0.15

    def test_item_correlation_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            generate_maxdiff_data(
                n_items=3,
                subset_size=2,
                item_correlation=np.eye(4),  # wrong size
            )

    def test_item_correlation_nonsymmetric_raises(self):
        bad = np.array([[1.0, 0.5, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            generate_maxdiff_data(n_items=3, subset_size=2, item_correlation=bad)

    def test_item_correlation_diagonal_not_one_raises(self):
        bad = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="ones on the diagonal"):
            generate_maxdiff_data(n_items=3, subset_size=2, item_correlation=bad)

    def test_item_correlation_not_psd_raises(self):
        # Invalid correlation matrix: not PSD
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        with pytest.raises(ValueError, match="positive semi-definite"):
            generate_maxdiff_data(n_items=3, subset_size=2, item_correlation=bad)


class TestGenerateMaxdiffConjointData:
    """Branch coverage for generate_maxdiff_conjoint_data validation paths."""

    def test_subset_size_too_small_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            generate_maxdiff_conjoint_data(n_items=5, subset_size=1)

    def test_subset_size_exceeds_auto_items_raises(self):
        """subset_size > n_items when item_attributes is None must raise."""
        with pytest.raises(ValueError, match="cannot exceed"):
            generate_maxdiff_conjoint_data(n_items=3, subset_size=5)

    def test_subset_size_exceeds_provided_attrs_raises(self):
        """subset_size > len(item_attributes) when attrs are supplied must raise."""
        attrs = pd.DataFrame(
            {"brand": ["A", "B"], "price": [0.2, 0.8], "quality": [0.1, -0.1]},
            index=pd.Index(["item_0", "item_1"], name="item_id"),
        )
        with pytest.raises(ValueError, match="cannot exceed"):
            generate_maxdiff_conjoint_data(
                item_attributes=attrs,
                utility_formula="~ 0 + C(brand) + price + quality",
                subset_size=5,
            )

    def test_items_inferred_from_provided_attrs(self):
        """When item_attributes is given and items=None, items come from the index."""
        attrs = pd.DataFrame(
            {
                "brand": ["A", "B", "C"],
                "price": [0.2, 0.5, 0.8],
                "quality": [0.1, 0.0, -0.1],
            },
            index=pd.Index(["x", "y", "z"], name="item_id"),
        )
        task_df, _, gt = generate_maxdiff_conjoint_data(
            item_attributes=attrs,
            utility_formula="~ 0 + C(brand) + price + quality",
            subset_size=2,
            n_respondents=3,
            n_tasks_per_resp=2,
            random_seed=1,
        )
        assert gt["items"] == ["x", "y", "z"]
        assert set(task_df["item_id"].unique()) <= {"x", "y", "z"}

    def test_true_betas_unknown_key_raises(self):
        """true_betas referencing a non-existent feature must raise."""
        with pytest.raises(ValueError, match="not in expanded features"):
            generate_maxdiff_conjoint_data(
                n_items=4,
                subset_size=2,
                true_betas={"not_a_feature": 1.0},
                random_seed=0,
            )

    def test_true_betas_override_applied(self):
        """true_betas entries must overwrite the drawn population betas."""
        _task_df, _, gt = generate_maxdiff_conjoint_data(
            n_items=4,
            subset_size=2,
            n_respondents=3,
            n_tasks_per_resp=2,
            true_betas={"price": -2.0, "quality": 3.0},
            random_seed=0,
        )
        price_idx = gt["feature_names"].index("price")
        quality_idx = gt["feature_names"].index("quality")
        assert gt["betas"][price_idx] == pytest.approx(-2.0)
        assert gt["betas"][quality_idx] == pytest.approx(3.0)

    def test_random_attributes_unknown_raises(self):
        """random_attributes containing an unknown feature name must raise."""
        with pytest.raises(ValueError, match="not in expanded features"):
            generate_maxdiff_conjoint_data(
                n_items=4,
                subset_size=2,
                random_attributes=["no_such_feature"],
                random_seed=0,
            )

    def test_random_attributes_subset_respected(self):
        """Only the named random_attributes columns should receive respondent noise."""
        _task_df, _, gt = generate_maxdiff_conjoint_data(
            n_items=6,
            subset_size=2,
            n_respondents=5,
            n_tasks_per_resp=2,
            random_attributes=["price"],
            random_seed=2,
        )
        assert gt["random_attributes"] == ["price"]
        price_idx = gt["feature_names"].index("price")
        # Population beta must differ from at least one respondent on price.
        pop_price = gt["betas"][price_idx]
        resp_prices = gt["respondent_betas"][:, price_idx]
        assert not np.all(resp_prices == pop_price)


# ---------------------------------------------------------------------------
# MaxDiffMixedLogit.sample — convenience wrapper
# ---------------------------------------------------------------------------


class TestSampleConvenienceWrapper:
    def test_sample_runs_and_populates_idata(self, small_maxdiff):
        """sample() must run prior predictive, fit, and posterior predictive."""
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        result = model.sample(
            fit_kwargs=FAST_FIT_KWARGS,
            sample_prior_predictive_kwargs={"samples": 5},
        )
        # sample() returns self for chaining.
        assert result is model
        assert model.idata is not None
        assert "prior_predictive" in model.idata.groups()
        assert "posterior" in model.idata.groups()
        assert "posterior_predictive" in model.idata.groups()

    def test_sample_builds_model_if_needed(self, small_maxdiff):
        """sample() must build the model automatically when not yet built."""
        task_df, items, _ = small_maxdiff
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        assert not hasattr(model, "model")
        model.sample(
            fit_kwargs=FAST_FIT_KWARGS,
            sample_prior_predictive_kwargs={"samples": 5},
        )
        assert hasattr(model, "model")


# ---------------------------------------------------------------------------
# Full LKJ covariance (HB-MaxDiff)
# ---------------------------------------------------------------------------


@pytest.fixture
def full_cov_fixture():
    """Small 4-item MaxDiff design for full-covariance tests."""
    return _small_maxdiff_df(
        n_respondents=5, n_items=4, n_tasks_per_resp=4, subset_size=3
    )


@pytest.fixture
def fitted_full_cov_model(full_cov_fixture):
    """Fitted full-covariance model."""
    task_df, items, _ = full_cov_fixture
    model = MaxDiffMixedLogit(
        task_df=task_df, items=items, random_intercepts=True, full_covariance=True
    )
    model.fit(random_seed=42, **FAST_FIT_KWARGS)
    return task_df, items, model


class TestFullCovariance:
    def test_build_smoke(self, full_cov_fixture):
        """Model builds and all expected variables/coords are present."""
        task_df, items, _ = full_cov_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df, items=items, random_intercepts=True, full_covariance=True
        )
        model.build_model()

        named = model.model.named_vars
        for var in ("chol_cov", "chol_L", "corr_matrix", "item_stds"):
            assert var in named, f"expected '{var}' in named_vars"

        assert "items_bis" in model.coords
        assert list(model.coords["items_bis"]) == items

    def test_logp_finite(self, full_cov_fixture):
        """Log-probability at the initial point must be finite."""
        task_df, items, _ = full_cov_fixture
        model = MaxDiffMixedLogit(
            task_df=task_df, items=items, random_intercepts=True, full_covariance=True
        )
        model.build_model()
        logp_fn = model.model.compile_logp()
        logp_val = float(logp_fn(model.model.initial_point()))
        assert np.isfinite(logp_val)

    def test_partworths_raises(self, full_cov_fixture):
        """full_covariance=True in part-worths mode must raise ValueError."""
        task_df, items, _ = full_cov_fixture
        attrs = pd.DataFrame({"price": np.linspace(0, 1, len(items))}, index=items)
        with pytest.raises(ValueError, match="part-worths mode"):
            MaxDiffMixedLogit(
                task_df=task_df,
                items=items,
                item_attributes=attrs,
                utility_formula="~ 0 + price",
                full_covariance=True,
            )

    def test_no_random_intercepts_raises(self, full_cov_fixture):
        """full_covariance=True without random_intercepts must raise ValueError."""
        task_df, items, _ = full_cov_fixture
        with pytest.raises(ValueError, match="random_intercepts=True"):
            MaxDiffMixedLogit(
                task_df=task_df,
                items=items,
                random_intercepts=False,
                full_covariance=True,
            )

    def test_save_load_roundtrip(self, fitted_full_cov_model, tmp_path):
        """full_covariance and lkj_eta must survive a save/load cycle."""
        _task_df, _items, model = fitted_full_cov_model
        path = tmp_path / "full_cov_model.nc"
        model.save(str(path))

        loaded = MaxDiffMixedLogit.load(str(path))
        assert loaded.full_covariance is True
        assert loaded.lkj_eta == pytest.approx(2.0)

    def test_new_respondent_population_draw_shape(self, fitted_full_cov_model):
        """Population draws for new respondents must have shape (C, D, n_new, I)."""
        _task_df, items, model = fitted_full_cov_model
        posterior = model.idata["posterior"]
        rng = np.random.default_rng(0)
        n_new = 3
        draws = model._draw_new_respondent_utilities(posterior, n_new=n_new, rng=rng)
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        n_items = len(items)
        assert draws.shape == (n_chains, n_draws, n_new, n_items)

    def test_new_respondent_uses_chol_L(self, fitted_full_cov_model):
        """Full-covariance draws must use chol_L, producing cross-item correlation."""
        _task_df, items, model = fitted_full_cov_model
        posterior = model.idata["posterior"]
        assert "chol_L" in posterior

        rng = np.random.default_rng(7)
        draws = model._draw_new_respondent_utilities(posterior, n_new=20, rng=rng)
        # Cross-item covariance of draws should not be exactly diagonal —
        # verify at least one off-diagonal element has non-trivial magnitude.
        # Flatten chain/draw into one axis: (C*D*20, I)
        flat = draws.reshape(-1, len(items))
        cov = np.cov(flat, rowvar=False)
        off_diag = cov[np.triu_indices(len(items), k=1)]
        # At least one off-diagonal covariance should be non-negligible
        assert np.any(np.abs(off_diag) > 1e-6)

    def test_items_bis_coord_absent_when_diagonal(self, full_cov_fixture):
        """Default diagonal model must NOT have an items_bis coord."""
        task_df, items, _ = full_cov_fixture
        model = MaxDiffMixedLogit(task_df=task_df, items=items)
        model.build_model()
        assert "items_bis" not in model.coords

    @pytest.mark.slow
    def test_correlation_matrix_recovered(self):
        """Posterior mean of corr_matrix should be close to the ground truth."""
        corr_true = np.array(
            [
                [1.0, 0.7, 0.2, 0.0, 0.0],
                [0.7, 1.0, 0.2, 0.0, 0.0],
                [0.2, 0.2, 1.0, 0.6, 0.1],
                [0.0, 0.0, 0.6, 1.0, 0.1],
                [0.0, 0.0, 0.1, 0.1, 1.0],
            ]
        )
        task_df, ground_truth = generate_maxdiff_data(
            n_respondents=300,  # enough for covariance
            n_items=5,  # >3 is key
            n_tasks_per_resp=10,  # repeated exposure
            subset_size=3,  # CRITICAL: joint comparisons
            sigma_respondent=1.0,
            item_correlation=corr_true,
            random_seed=7,
        )
        model = MaxDiffMixedLogit(
            task_df=task_df,
            items=ground_truth["items"],
            full_covariance=True,
        )
        model.fit(
            draws=500,
            tune=1000,
            chains=2,
            target_accept=0.9,
            nuts_sampler="pymc",
            random_seed=7,
        )
        corr_post = (
            model.idata["posterior"]["corr_matrix"].mean(("chain", "draw")).values
        )
        assert corr_post[0, 1] > 0.3
        assert corr_post[0, 1] > corr_post[0, 3]
