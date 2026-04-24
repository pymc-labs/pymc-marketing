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
r"""MaxDiff (Best-Worst Scaling) hierarchical Bayesian model.

Implements the Louviere sequential best-worst model:

    P(best, worst | subset) = P(best | subset) * P(worst | subset \\ {best})

where the worst pick uses sign-flipped utilities. Item-level utilities are
estimated with optional per-respondent random intercepts (hierarchical /
HB-MaxDiff). The reference item's utility is pinned to zero for identification.
"""

import json
import warnings
from typing import Any, Self, TypedDict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.util import RandomState
from pymc_extras.prior import Prior

from pymc_marketing.model_builder import ModelBuilder, create_sample_kwargs
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.version import __version__

NEG_INF = -1e9


def _softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax using max-subtraction."""
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x_shift)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _sample_categorical(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a category index per leading batch from a probability array.

    ``p`` has shape ``(..., K)``; the return has shape ``(...)``.
    """
    cumulative = np.cumsum(p, axis=-1)
    # Draw one uniform per leading-batch cell
    u = rng.random(cumulative.shape[:-1])
    return np.sum(cumulative < u[..., None], axis=-1).astype(np.int64)


class MaxDiffArrays(TypedDict):
    """Preprocessed arrays ready for the MaxDiff likelihood.

    Attributes
    ----------
    item_idx : np.ndarray
        Shape ``(T, K_max)`` int64. Items shown in each task as indices into the
        item pool, padded with the reference item's index where ``mask`` is False.
    mask : np.ndarray
        Shape ``(T, K_max)`` bool. True where the position is a real shown item.
    best_pos : np.ndarray
        Shape ``(T,)`` int64. Position (0..K_max-1) of the best-chosen item.
    worst_pos : np.ndarray
        Shape ``(T,)`` int64. Position of the worst-chosen item.
    resp_idx : np.ndarray
        Shape ``(T,)`` int64. Respondent index for each task.
    n_tasks : int
        Number of tasks ``T``.
    n_respondents : int
        Number of unique respondents.
    n_items : int
        Size of the full item pool.
    k_max : int
        Maximum subset size across all tasks.
    item_to_idx : dict[str, int]
        Mapping from item name to integer index.
    respondent_to_idx : dict[Any, int]
        Mapping from respondent identifier to integer index.
    """

    item_idx: np.ndarray
    mask: np.ndarray
    best_pos: np.ndarray
    worst_pos: np.ndarray
    resp_idx: np.ndarray
    n_tasks: int
    n_respondents: int
    n_items: int
    k_max: int
    item_to_idx: dict[str, int]
    respondent_to_idx: dict[Any, int]


def prepare_maxdiff_data(
    task_df: pd.DataFrame,
    items: list[str],
    respondent_id: str = "respondent_id",
    task_id: str = "task_id",
    item_col: str = "item_id",
    best_col: str = "is_best",
    worst_col: str = "is_worst",
    reference_item: str | None = None,
) -> MaxDiffArrays:
    """Reshape long-format MaxDiff data into padded arrays for the likelihood.

    Each row of ``task_df`` represents one shown item within one task. Tasks
    may show different numbers of items (ragged subset sizes are padded to
    ``K_max`` with the reference item; ``mask`` marks which positions are real).

    Parameters
    ----------
    task_df : pd.DataFrame
        Long-format data with one row per (respondent, task, item) triple.
        Must contain the five columns named by ``respondent_id``, ``task_id``,
        ``item_col``, ``best_col``, ``worst_col``.
    items : list[str]
        Full item pool. Defines the ``items`` coord and the index mapping.
    respondent_id, task_id, item_col, best_col, worst_col : str
        Column names in ``task_df``.
    reference_item : str, optional
        Item whose utility is pinned to 0 for identification. Defaults to
        ``items[-1]``.

    Returns
    -------
    MaxDiffArrays
        TypedDict of padded arrays and metadata.

    Raises
    ------
    ValueError
        If a task lacks exactly one best or worst pick, best == worst within a
        task, items repeat within a task, or any item is outside the pool.
    """
    required = {respondent_id, task_id, item_col, best_col, worst_col}
    missing = required - set(task_df.columns)
    if missing:
        raise ValueError(f"task_df is missing required columns: {sorted(missing)}")

    if reference_item is None:
        reference_item = items[-1]
    if reference_item not in items:
        raise ValueError(f"reference_item '{reference_item}' is not in the item pool.")

    item_to_idx = {item: i for i, item in enumerate(items)}
    ref_idx = item_to_idx[reference_item]

    unknown = set(task_df[item_col].unique()) - set(items)
    if unknown:
        raise ValueError(
            f"Items in task_df not found in the item pool: {sorted(unknown)}"
        )

    respondents = list(pd.unique(task_df[respondent_id]))
    respondent_to_idx = {r: i for i, r in enumerate(respondents)}

    grouped = task_df.groupby([respondent_id, task_id], sort=False)
    task_keys = list(grouped.groups.keys())
    n_tasks = len(task_keys)

    if n_tasks == 0:
        raise ValueError("task_df contains no tasks.")

    k_max = grouped.size().max()

    item_idx = np.full((n_tasks, k_max), ref_idx, dtype=np.int64)
    mask = np.zeros((n_tasks, k_max), dtype=bool)
    best_pos = np.empty(n_tasks, dtype=np.int64)
    worst_pos = np.empty(n_tasks, dtype=np.int64)
    resp_idx = np.empty(n_tasks, dtype=np.int64)

    for t, (key, group) in enumerate(grouped):
        resp_key, _ = key
        resp_idx[t] = respondent_to_idx[resp_key]

        group_items = group[item_col].tolist()
        if len(set(group_items)) != len(group_items):
            raise ValueError(f"Task {key} has duplicate items: {group_items}.")

        k = len(group_items)
        item_idx[t, :k] = [item_to_idx[x] for x in group_items]
        mask[t, :k] = True

        best_flags = group[best_col].to_numpy()
        worst_flags = group[worst_col].to_numpy()

        n_best = int(best_flags.sum())
        n_worst = int(worst_flags.sum())
        if n_best != 1:
            raise ValueError(
                f"Task {key} has {n_best} best picks; exactly one is required."
            )
        if n_worst != 1:
            raise ValueError(
                f"Task {key} has {n_worst} worst picks; exactly one is required."
            )

        bp = int(np.argmax(best_flags))
        wp = int(np.argmax(worst_flags))
        if bp == wp:
            raise ValueError(f"Task {key} has the same item as both best and worst.")

        best_pos[t] = bp
        worst_pos[t] = wp

    return MaxDiffArrays(
        item_idx=item_idx,
        mask=mask,
        best_pos=best_pos,
        worst_pos=worst_pos,
        resp_idx=resp_idx,
        n_tasks=n_tasks,
        n_respondents=len(respondents),
        n_items=len(items),
        k_max=int(k_max),
        item_to_idx=item_to_idx,
        respondent_to_idx=respondent_to_idx,
    )


class MaxDiffMixedLogit(ModelBuilder):
    r"""Hierarchical MaxDiff (Best-Worst Scaling) model.

    Estimates item-level utilities from best-worst choice data with optional
    per-respondent random intercepts. The likelihood is the Louviere sequential
    best-worst model:

    .. math::

        P(\\text{best}_t = b \\mid \\text{subset}_t) &= \\operatorname{softmax}(U)_b \\\\
        P(\\text{worst}_t = w \\mid \\text{subset}_t, b) &=
            \\operatorname{softmax}(-U_{\\setminus b})_w

    implemented as two ``pm.Categorical`` observed distributions so that
    ``pm.sample_posterior_predictive`` yields best/worst draws directly.

    Parameters
    ----------
    task_df : pd.DataFrame
        Long-format MaxDiff data; see :func:`prepare_maxdiff_data`.
    items : list[str]
        Full item pool. Defines the ``items`` coord.
    respondent_id : str, default "respondent_id"
        Column in ``task_df`` identifying respondents.
    task_id : str, default "task_id"
        Column identifying tasks (unique within respondent).
    item_col : str, default "item_id"
        Column naming the shown item (must be in ``items``).
    best_col : str, default "is_best"
        0/1 column flagging the best pick within each task.
    worst_col : str, default "is_worst"
        0/1 column flagging the worst pick.
    random_intercepts : bool, default True
        When True, each respondent draws item-level deviations from the
        population item utilities (HB-MaxDiff). When False, only population
        utilities are estimated.
    reference_item : str, optional
        Item pinned to utility 0 for identification. Defaults to ``items[-1]``.
    model_config : dict, optional
        Priors for ``beta_item_`` (population utilities) and ``sigma_item``
        (per-item heterogeneity scale).
    sampler_config : dict, optional
        Arguments passed to ``pm.sample``.
    non_centered : bool, default True
        Non-centered parameterisation for the respondent-level deviations.

    Notes
    -----
    Input format example::

        respondent_id  task_id  item_id  is_best  is_worst
        r1             1        apple    0        0
        r1             1        banana   1        0
        r1             1        cherry   0        1
        r1             1        date     0        0
        r1             2        apple    0        1
        ...

    Each ``(respondent_id, task_id)`` group must contain exactly one row with
    ``is_best == 1`` and one with ``is_worst == 1``, and the two must differ.
    Subset sizes may vary across tasks; they are padded to ``K_max`` internally.

    The default reference item is ``items[-1]``. Only item-utility *contrasts*
    against the reference are identified; absolute levels are not.
    """

    _model_type = "MaxDiff Mixed Logit"
    version = "0.1.0"

    def __init__(
        self,
        task_df: pd.DataFrame,
        items: list[str],
        respondent_id: str = "respondent_id",
        task_id: str = "task_id",
        item_col: str = "item_id",
        best_col: str = "is_best",
        worst_col: str = "is_worst",
        random_intercepts: bool = True,
        reference_item: str | None = None,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        non_centered: bool = True,
    ):
        self.task_df = task_df
        self.items = list(items)
        self.respondent_id = respondent_id
        self.task_id = task_id
        self.item_col = item_col
        self.best_col = best_col
        self.worst_col = worst_col
        self.random_intercepts = random_intercepts
        self.reference_item = (
            reference_item if reference_item is not None else items[-1]
        )
        self.non_centered = non_centered

        if self.reference_item not in self.items:
            raise ValueError(
                f"reference_item '{self.reference_item}' is not in the item pool."
            )

        model_config = model_config or {}
        model_config = parse_model_config(model_config)

        super().__init__(model_config=model_config, sampler_config=sampler_config)

    @property
    def default_model_config(self) -> dict:
        """Default priors for item utilities and respondent heterogeneity."""
        return {
            "beta_item_": Prior("Normal", mu=0, sigma=2, dims="items"),
            "sigma_item": Prior("HalfNormal", sigma=1, dims="items"),
            "z_item": Prior("Normal", mu=0, sigma=1, dims=("respondents", "items")),
        }

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {
            "nuts_sampler": "numpyro",
            "idata_kwargs": {"log_likelihood": True},
        }

    @property
    def output_var(self) -> str:
        """Primary observed variable name."""
        return "best_pick"

    @property
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        result: dict[str, int | float | dict] = {
            "beta_item_": self.model_config["beta_item_"].to_dict(),
        }
        if self.random_intercepts:
            result["sigma_item"] = self.model_config["sigma_item"].to_dict()
            result["z_item"] = self.model_config["z_item"].to_dict()
        return result

    def preprocess_model_data(self, task_df: pd.DataFrame) -> MaxDiffArrays:
        """Run :func:`prepare_maxdiff_data` and cache its outputs on the model."""
        arrays = prepare_maxdiff_data(
            task_df=task_df,
            items=self.items,
            respondent_id=self.respondent_id,
            task_id=self.task_id,
            item_col=self.item_col,
            best_col=self.best_col,
            worst_col=self.worst_col,
            reference_item=self.reference_item,
        )

        self.arrays = arrays
        self.coords = {
            "items": self.items,
            "respondents": list(arrays["respondent_to_idx"].keys()),
            "tasks": list(range(arrays["n_tasks"])),
            "positions": list(range(arrays["k_max"])),
        }
        return arrays

    def build_model(self, **kwargs) -> None:
        """Build the PyMC model using the cached ``task_df``."""
        arrays = self.preprocess_model_data(self.task_df)
        self.model = self.make_model(arrays)

    def make_model(
        self,
        arrays: MaxDiffArrays,
        observed: bool = True,
    ) -> pm.Model:
        """Build the MaxDiff PyMC model.

        Parameters
        ----------
        arrays : MaxDiffArrays
            Preprocessed padded-plus-mask representation of the tasks.
        observed : bool, default True
            Whether to attach observed data to the Categorical likelihoods.
            When False, the model can be used for forward simulation.

        Returns
        -------
        pm.Model
            Model with ``beta_item``, optional ``beta_item_r``, and two
            observed ``Categorical`` likelihoods ``best_pick`` / ``worst_pick``.
        """
        ref_idx = self.items.index(self.reference_item)

        with pm.Model(coords=self.coords) as model:
            item_idx = pm.Data(
                "item_idx", arrays["item_idx"], dims=("tasks", "positions")
            )
            mask = pm.Data("mask", arrays["mask"], dims=("tasks", "positions"))
            best_pos = pm.Data("best_pos", arrays["best_pos"], dims="tasks")
            worst_pos = pm.Data("worst_pos", arrays["worst_pos"], dims="tasks")
            resp_idx = pm.Data("resp_idx", arrays["resp_idx"], dims="tasks")

            beta_item_ = self.model_config["beta_item_"].create_variable("beta_item_")
            beta_item = pm.Deterministic(
                "beta_item", pt.set_subtensor(beta_item_[ref_idx], 0.0), dims="items"
            )

            if self.random_intercepts:
                sigma_item = self.model_config["sigma_item"].create_variable(
                    "sigma_item"
                )
                if self.non_centered:
                    z_item = self.model_config["z_item"].create_variable("z_item")
                    beta_item_r = pm.Deterministic(
                        "beta_item_r",
                        beta_item[None, :] + z_item * sigma_item[None, :],
                        dims=("respondents", "items"),
                    )
                else:
                    beta_item_r = pm.Normal(
                        "beta_item_r",
                        mu=beta_item[None, :],
                        sigma=sigma_item[None, :],
                        dims=("respondents", "items"),
                    )
                U = beta_item_r[resp_idx[:, None], item_idx]
            else:
                U = beta_item[item_idx]

            U = pm.Deterministic("U", U, dims=("tasks", "positions"))

            # Best pick: masked softmax over the subset
            U_best_masked = pt.where(mask, U, NEG_INF)
            p_best = pm.Deterministic(
                "p_best",
                pm.math.softmax(U_best_masked, axis=1),
                dims=("tasks", "positions"),
            )

            # Worst pick: sign-flipped utility, excluding the chosen best
            rows = pt.arange(arrays["n_tasks"])
            U_worst_signflip = pt.where(mask, -U, NEG_INF)
            U_worst_masked = pt.set_subtensor(U_worst_signflip[rows, best_pos], NEG_INF)
            p_worst = pm.Deterministic(
                "p_worst",
                pm.math.softmax(U_worst_masked, axis=1),
                dims=("tasks", "positions"),
            )

            if observed:
                pm.Categorical("best_pick", p=p_best, observed=best_pos, dims="tasks")
                pm.Categorical(
                    "worst_pick", p=p_worst, observed=worst_pos, dims="tasks"
                )
            else:
                pm.Categorical("best_pick", p=p_best, dims="tasks")
                pm.Categorical("worst_pick", p=p_worst, dims="tasks")

        return model

    def create_idata_attrs(self) -> dict[str, str]:
        """Serialise init kwargs so the model can be reloaded from idata."""
        attrs = super().create_idata_attrs()
        # task_df is stored in the fit_data group, not in attrs
        attrs["task_df"] = json.dumps("Placeholder for DataFrame")
        attrs["items"] = json.dumps(self.items)
        attrs["respondent_id"] = json.dumps(self.respondent_id)
        attrs["task_id"] = json.dumps(self.task_id)
        attrs["item_col"] = json.dumps(self.item_col)
        attrs["best_col"] = json.dumps(self.best_col)
        attrs["worst_col"] = json.dumps(self.worst_col)
        attrs["random_intercepts"] = json.dumps(self.random_intercepts)
        attrs["reference_item"] = json.dumps(self.reference_item)
        attrs["non_centered"] = json.dumps(self.non_centered)
        return attrs

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        """Rehydrate init kwargs from serialised idata attrs."""
        return {
            "task_df": pd.DataFrame(),  # replaced by build_from_idata
            "items": json.loads(attrs["items"]),
            "respondent_id": json.loads(attrs["respondent_id"]),
            "task_id": json.loads(attrs["task_id"]),
            "item_col": json.loads(attrs["item_col"]),
            "best_col": json.loads(attrs["best_col"]),
            "worst_col": json.loads(attrs["worst_col"]),
            "random_intercepts": json.loads(attrs["random_intercepts"]),
            "reference_item": json.loads(attrs["reference_item"]),
            "non_centered": json.loads(attrs["non_centered"]),
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "sampler_config": json.loads(attrs["sampler_config"]),
        }

    def _create_fit_data(self) -> xr.Dataset:
        """Serialise ``task_df`` so :meth:`load` can reconstruct the model."""
        df_xr = self.task_df.reset_index(drop=True).to_xarray()
        df_xr = df_xr.rename({"index": "row"})
        return df_xr

    def build_from_idata(self, idata: az.InferenceData) -> None:
        """Rebuild the PyMC model from a loaded InferenceData."""
        self.task_df = idata["fit_data"].to_dataframe().reset_index(drop=True)
        if not hasattr(self, "model"):
            self.build_model()

    def sample_prior_predictive(
        self,
        task_df: pd.DataFrame | None = None,
        samples: int = 500,
        extend_idata: bool = True,
        **kwargs,
    ) -> az.InferenceData:
        """Sample from the prior predictive distribution."""
        if task_df is not None:
            self.task_df = task_df

        if not hasattr(self, "model"):
            self.build_model()

        with self.model:
            prior_pred = pm.sample_prior_predictive(samples, **kwargs)
            prior_pred["prior"].attrs["pymc_marketing_version"] = __version__
            prior_pred["prior_predictive"].attrs["pymc_marketing_version"] = __version__
            self.set_idata_attrs(prior_pred)

        if extend_idata:
            if self.idata is not None:
                self.idata.extend(prior_pred, join="right")
            else:
                self.idata = prior_pred

        return prior_pred

    def fit(
        self,
        task_df: pd.DataFrame | None = None,
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        **kwargs,
    ) -> az.InferenceData:
        """Fit the model via NUTS and attach the result to ``self.idata``."""
        if task_df is not None:
            self.task_df = task_df

        if not hasattr(self, "model"):
            self.build_model()

        sampler_kwargs = create_sample_kwargs(
            self.sampler_config,
            progressbar,
            random_seed,
            **kwargs,
        )

        with self.model:
            idata = pm.sample(**sampler_kwargs)

        if self.idata:
            self.idata = self.idata.copy()
            self.idata.extend(idata, join="right")
        else:
            self.idata = idata

        self.idata["posterior"].attrs["pymc_marketing_version"] = __version__

        if "fit_data" in self.idata:
            del self.idata.fit_data

        fit_data = self._create_fit_data()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=fit_data)

        self.set_idata_attrs(self.idata)
        return self.idata

    def sample_posterior_predictive(
        self,
        task_df: pd.DataFrame | None = None,
        extend_idata: bool = True,
        **kwargs,
    ) -> az.InferenceData:
        """Sample from the posterior predictive distribution.

        When ``task_df`` is provided, the model's data containers are updated
        via ``pm.set_data`` before sampling. ``worst_pick`` is drawn conditional
        on the observed ``best_pos`` — for fully generative counterfactual
        simulation use :meth:`predict_choices`.
        """
        if task_df is not None:
            arrays = self.preprocess_model_data(task_df)
            with self.model:
                pm.set_data(
                    {
                        "item_idx": arrays["item_idx"],
                        "mask": arrays["mask"],
                        "best_pos": arrays["best_pos"],
                        "worst_pos": arrays["worst_pos"],
                        "resp_idx": arrays["resp_idx"],
                    }
                )

        if self.idata is None:
            raise RuntimeError("self.idata must be initialized before extending.")

        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata,
                var_names=["best_pick", "worst_pick", "p_best", "p_worst"],
                **kwargs,
            )

        if extend_idata:
            self.idata.extend(post_pred, join="right")

        return post_pred

    def predict_choices(
        self,
        task_df: pd.DataFrame,
        random_seed: RandomState | None = None,
    ) -> xr.Dataset:
        """Fully generative (best, worst) simulation under a new task design.

        Unlike :meth:`sample_posterior_predictive`, this method does *not*
        condition the worst softmax on the observed best pick. For each
        posterior draw, it:

        1. Computes per-task utilities from ``beta_item_r`` (or ``beta_item``).
        2. Samples a best position from ``softmax(U)``.
        3. Samples a worst position from ``softmax(-U)`` with the sampled best
           position excluded.

        The resulting ``worst_pick`` is conditional on the freshly sampled
        ``best_pick`` — the correct generative joint distribution.

        Parameters
        ----------
        task_df : pd.DataFrame
            New long-format task data. ``is_best`` / ``is_worst`` columns may
            be dummy values; they are ignored for prediction.
        random_seed : RandomState, optional
            Seed for the numpy Generator used to sample best and worst.

        Returns
        -------
        xr.Dataset
            ``posterior_predictive``-shaped dataset with ``best_pick`` and
            ``worst_pick`` variables of shape ``(chain, draw, tasks)`` and
            ``p_best`` / ``p_worst`` of shape ``(chain, draw, tasks, positions)``.

        Notes
        -----
        Respondents in ``task_df`` must be a subset of the training
        respondents when ``random_intercepts=True``; posterior draws of
        ``beta_item_r`` are indexed by training respondent.
        """
        if self.idata is None:
            raise RuntimeError(
                "predict_choices requires a fitted model. Call .fit() first."
            )

        arrays = prepare_maxdiff_data(
            task_df=task_df,
            items=self.items,
            respondent_id=self.respondent_id,
            task_id=self.task_id,
            item_col=self.item_col,
            best_col=self.best_col,
            worst_col=self.worst_col,
            reference_item=self.reference_item,
        )

        posterior = self.idata["posterior"]
        if self.random_intercepts:
            training_respondents = list(self.coords["respondents"])
            resp_name_to_training_idx = {
                r: i for i, r in enumerate(training_respondents)
            }
            try:
                pred_resp_training_idx = np.array(
                    [
                        resp_name_to_training_idx[r]
                        for r in arrays["respondent_to_idx"].keys()
                    ],
                    dtype=np.int64,
                )
            except KeyError as e:
                raise ValueError(
                    f"Respondent {e.args[0]!r} in task_df was not in the "
                    "training data. predict_choices with random_intercepts=True "
                    "requires known respondents."
                ) from e
            # beta_item_r: (chain, draw, respondents, items)
            beta_r = posterior["beta_item_r"].values
            # Map prediction-time resp_idx (into arrays respondents) to
            # training respondent index.
            training_resp_per_task = pred_resp_training_idx[arrays["resp_idx"]]
            # Gather U: (chain, draw, T, K_max)
            # beta_r[..., training_resp_per_task, :] -> (chain, draw, T, I)
            U_respondent = beta_r[:, :, training_resp_per_task, :]
            # item_idx: (T, K_max); need (chain, draw, T, K_max).
            U = np.take_along_axis(
                U_respondent,
                np.broadcast_to(
                    arrays["item_idx"][None, None, :, :],
                    U_respondent.shape[:2] + arrays["item_idx"].shape,
                ),
                axis=-1,
            )
        else:
            # beta_item: (chain, draw, items)
            beta = posterior["beta_item"].values
            U = beta[:, :, arrays["item_idx"]]

        chain, draw, T, k_max = U.shape
        mask = arrays["mask"]

        U_best_masked = np.where(mask[None, None, :, :], U, NEG_INF)
        p_best = _softmax_stable(U_best_masked, axis=-1)

        rng = (
            random_seed
            if isinstance(random_seed, np.random.Generator)
            else np.random.default_rng(random_seed)
        )
        best_samples = _sample_categorical(p_best, rng)  # (chain, draw, T)

        U_worst_signflip = np.where(mask[None, None, :, :], -U, NEG_INF)
        rows = np.arange(T)
        # Zero-out sampled best position: indices (chain, draw, T) -> positions
        flat_tasks = np.broadcast_to(rows[None, None, :], (chain, draw, T))
        chain_idx = np.arange(chain)[:, None, None]
        draw_idx = np.arange(draw)[None, :, None]
        U_worst_masked = U_worst_signflip.copy()
        U_worst_masked[chain_idx, draw_idx, flat_tasks, best_samples] = NEG_INF
        p_worst = _softmax_stable(U_worst_masked, axis=-1)
        worst_samples = _sample_categorical(p_worst, rng)

        result = xr.Dataset(
            {
                "best_pick": (("chain", "draw", "tasks"), best_samples),
                "worst_pick": (("chain", "draw", "tasks"), worst_samples),
                "p_best": (("chain", "draw", "tasks", "positions"), p_best),
                "p_worst": (("chain", "draw", "tasks", "positions"), p_worst),
            },
            coords={
                "chain": posterior["chain"].values,
                "draw": posterior["draw"].values,
                "tasks": np.arange(T),
                "positions": np.arange(k_max),
            },
        )
        return result

    def sample(
        self,
        sample_prior_predictive_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        sample_posterior_predictive_kwargs: dict | None = None,
    ) -> Self:
        """Run prior predictive, fit, and posterior predictive in sequence."""
        sample_prior_predictive_kwargs = sample_prior_predictive_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        sample_posterior_predictive_kwargs = sample_posterior_predictive_kwargs or {}

        if not hasattr(self, "model"):
            self.build_model()

        self.sample_prior_predictive(
            extend_idata=True, **sample_prior_predictive_kwargs
        )
        self.fit(extend_idata=True, **fit_kwargs)
        self.sample_posterior_predictive(
            extend_idata=True, **sample_posterior_predictive_kwargs
        )
        return self
