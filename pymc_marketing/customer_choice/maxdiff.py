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
from typing import Any, Literal, Self, TypedDict

import arviz as az
import numpy as np
import pandas as pd
import patsy
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


def _sample_categorical_from_u(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample a category index from ``p`` using pre-drawn uniforms ``u``.

    ``p`` has shape ``(..., K)``; ``u`` has shape ``(...)`` matching the leading
    batch of ``p``. Returns an integer array of shape ``(...)``. Splitting the
    uniform draw from the categorical-lookup logic lets us pre-allocate the
    full uniform tensor and slice it across draw batches, guaranteeing
    bit-identical output regardless of ``draw_batch_size``.
    """
    cumulative = np.cumsum(p, axis=-1)
    return np.sum(cumulative < u[..., None], axis=-1).astype(np.int64)


def _sample_categorical(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a category index per leading batch from a probability array.

    ``p`` has shape ``(..., K)``; the return has shape ``(...)``.
    """
    u = rng.random(p.shape[:-1])
    return _sample_categorical_from_u(p, u)


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
        task, items repeat within a task, a task shows fewer than 2 items, or
        any item is outside the pool.
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

    sizes = grouped.size()
    too_small = sizes[sizes < 2]
    if len(too_small) > 0:
        example_key = too_small.index[0]
        example_k = int(too_small.iloc[0])
        raise ValueError(
            "Every (respondent, task) must show at least 2 items for "
            f"best-worst scaling. Found {len(too_small)} task(s) with <2 items, "
            f"e.g. {example_key!r} with {example_k} item(s)."
        )

    k_max = sizes.max()

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
    item_attributes : pd.DataFrame, optional
        One row per item, with the item name as the index and one column per
        attribute. When provided together with ``utility_formula``, switches
        the model into **part-worths mode**: utilities become
        :math:`U_i = X_i^\\top \\beta_{\\mathrm{feat}}` where :math:`X` is the
        patsy-expanded design matrix. Extrapolates naturally to new items via
        their attributes. Must cover every item in ``items``.
    utility_formula : str, optional
        Patsy formula describing the attribute contribution to utility,
        e.g. ``"~ 0 + C(brand) + price + quality"``. Required iff
        ``item_attributes`` is given. Use a leading ``0 +`` (no intercept) so
        the model is identified without a reference item.
    random_attributes : list[str], optional
        Names of patsy-expanded feature columns that should vary across
        respondents (respondent part-worths). Remaining features are treated
        as population-level fixed effects. Only meaningful in part-worths mode;
        ignored otherwise. Defaults to an empty list (pure fixed part-worths).

        .. note::

            Other customer-choice models in this package use Wilkinson pipe
            notation ``"~ covariate | random_covariate"`` to declare random
            coefficients. MaxDiff deliberately diverges: there is no
            per-alternative equation structure here (the same attributes
            describe every item), so the pipe formula is ambiguous. An
            explicit list is cleaner and less error-prone.

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
    Each task must show **at least two items**. Subset sizes may vary across
    tasks; they are padded to ``K_max`` internally.

    In the default (item-intercept) mode only item-utility *contrasts*
    against the reference item are identified; absolute levels are not.
    In part-worths mode ``reference_item`` / ``random_intercepts`` are
    ignored — identification comes from the no-intercept formula
    (``~ 0 + ...``) and respondent heterogeneity is controlled by
    ``random_attributes``.

    .. rubric:: Posterior predictive limitations

    The Louviere best-worst likelihood is **sequential**: worst is drawn from
    the remaining items *after* the best has been removed. In the PyMC graph
    this is implemented by masking the best position out of the worst-pick
    softmax using ``best_pos`` as a ``pm.Data`` node.

    :meth:`sample_posterior_predictive` therefore produces a **partially
    conditioned** joint:

    * ``best_pick`` is sampled correctly from ``softmax(U)``.
    * ``worst_pick`` is sampled from ``softmax(-U \\ {observed_best})``,
      i.e. it is still conditioned on the *observed* best position, not on
      the freshly sampled ``best_pick``.

    This makes the joint ``(best_pick, worst_pick)`` draws **incoherent for
    generative use** — the two picks may designate the same position.
    :meth:`sample_posterior_predictive` remains valid for **in-sample
    posterior predictive checks**: verifying that the model's worst-pick
    distribution is consistent with the data, given that the best pick was
    what was actually recorded.

    For any **counterfactual or out-of-sample** simulation use
    :meth:`predict_choices` (or :meth:`apply_intervention`), which samples
    the joint ``(best, worst)`` generatively — best first, then worst
    conditioned on the *sampled* best — producing a coherent joint draw.
    """

    _model_type = "MaxDiff Mixed Logit"
    version = "0.3.0"

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
        item_attributes: pd.DataFrame | None = None,
        utility_formula: str | None = None,
        random_attributes: list[str] | None = None,
        full_covariance: bool = False,
        lkj_eta: float = 2.0,
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

        # Part-worths configuration. Both-or-neither: item_attributes and
        # utility_formula must be provided together, or both omitted.
        has_attrs = item_attributes is not None
        has_formula = utility_formula is not None
        if has_attrs != has_formula:
            raise ValueError(
                "item_attributes and utility_formula must be provided together "
                "(or both omitted). Got item_attributes="
                f"{'set' if has_attrs else 'None'}, utility_formula="
                f"{'set' if has_formula else 'None'}."
            )

        self._is_partworths = has_attrs
        self.utility_formula = utility_formula
        self.item_attributes = item_attributes
        self.random_attributes: list[str] = list(random_attributes or [])

        if self._is_partworths:
            # In part-worths mode identification comes from the ``0 +`` in the
            # patsy formula, not a reference item; and per-respondent variation
            # is driven by ``random_attributes``, not ``random_intercepts``.
            self.random_intercepts = False
            self._prepare_design_matrix()
        else:
            self.X: np.ndarray | None = None
            self.feature_names: list[str] = []
            self._design_info: Any = None
            if self.random_attributes:
                raise ValueError(
                    "random_attributes is only valid when item_attributes and "
                    "utility_formula are supplied (part-worths mode)."
                )

        if self.reference_item not in self.items:
            raise ValueError(
                f"reference_item '{self.reference_item}' is not in the item pool."
            )

        if full_covariance and self._is_partworths:
            raise ValueError(
                "full_covariance is not supported in part-worths mode. "
                "Item correlations are already structured through the design matrix X. "
                "Use random_attributes for respondent-level variation."
            )
        if full_covariance and not self.random_intercepts:
            raise ValueError(
                "full_covariance=True requires random_intercepts=True. "
                "There are no per-respondent effects to correlate."
            )
        self.full_covariance = full_covariance
        self.lkj_eta = lkj_eta

        model_config = model_config or {}
        model_config = parse_model_config(model_config)

        super().__init__(model_config=model_config, sampler_config=sampler_config)

    def _prepare_design_matrix(self) -> None:
        """Align ``item_attributes`` to ``self.items`` and build ``self.X``.

        Populates ``self.X`` (``(n_items, n_features)``), ``self.feature_names``,
        and ``self._design_info`` so new-item extrapolation is possible by
        re-applying the saved ``DesignInfo`` to fresh attribute rows.
        """
        attrs = self.item_attributes
        if not isinstance(attrs, pd.DataFrame):
            raise TypeError(
                "item_attributes must be a pandas DataFrame indexed by item name."
            )
        missing = set(self.items) - set(attrs.index)
        if missing:
            raise ValueError(
                f"item_attributes is missing rows for {len(missing)} item(s): "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}."
            )
        aligned = attrs.loc[self.items]
        design = patsy.dmatrix(self.utility_formula, aligned, return_type="matrix")
        self._design_info = design.design_info
        self.X = np.asarray(design, dtype=float)
        self.feature_names = list(design.design_info.column_names)

        unknown = [rc for rc in self.random_attributes if rc not in self.feature_names]
        if unknown:
            raise ValueError(
                f"random_attributes {unknown} not found in expanded feature "
                f"names {self.feature_names}. Names must match exactly — "
                "patsy expands categoricals into multiple columns "
                "(e.g. 'C(brand)[T.B]')."
            )

    def transform_attributes(self, new_attrs: pd.DataFrame) -> np.ndarray:
        """Apply the fitted patsy formula to a new attribute frame.

        Use this to score hypothetical items that were not in the training
        pool. Returns a ``(n_rows, n_features)`` numpy array whose columns
        align with ``self.feature_names`` — multiplying by a posterior draw of
        ``beta_feat`` gives the posterior of the new items' utilities.

        Parameters
        ----------
        new_attrs : pd.DataFrame
            One row per hypothetical item, with the same attribute columns as
            the training ``item_attributes``.

        Raises
        ------
        RuntimeError
            If called on a non-part-worths model (no design info to apply).
        """
        if not self._is_partworths:
            raise RuntimeError(
                "transform_attributes is only available in part-worths mode "
                "(item_attributes + utility_formula must be set at __init__)."
            )
        return np.asarray(
            patsy.dmatrix(self._design_info, new_attrs, return_type="matrix")
        )

    def _ensure_best_worst_flags(self, task_df: pd.DataFrame) -> pd.DataFrame:
        """Return ``task_df`` with ``is_best`` / ``is_worst`` columns guaranteed.

        When **both** flag columns are absent, dummy flags are auto-generated
        (first item in each task marked best, last marked worst). This satisfies
        the :func:`prepare_maxdiff_data` column-presence check while keeping
        the flags out of the prediction logic.

        Parameters
        ----------
        task_df : pd.DataFrame
            Task design frame, possibly without ``is_best`` / ``is_worst``.

        Returns
        -------
        pd.DataFrame
            A copy of ``task_df`` with both flag columns present.

        Raises
        ------
        ValueError
            If exactly one of the two flag columns is present (partial labelling
            is ambiguous and likely a caller mistake).
        """
        has_best = self.best_col in task_df.columns
        has_worst = self.worst_col in task_df.columns
        if has_best and has_worst:
            return task_df
        if has_best != has_worst:
            present = self.best_col if has_best else self.worst_col
            absent = self.worst_col if has_best else self.best_col
            raise ValueError(
                f"Column '{present}' is present but '{absent}' is absent. "
                "Provide both flag columns or omit both (dummy flags are "
                "auto-generated when both are absent)."
            )
        # Both absent: auto-generate position-based dummy flags.
        df = task_df.copy()
        g = df.groupby([self.respondent_id, self.task_id], sort=False).cumcount()
        sizes = df.groupby([self.respondent_id, self.task_id], sort=False)[
            self.item_col
        ].transform("size")
        df[self.best_col] = (g == 0).astype(int)
        df[self.worst_col] = (g == sizes - 1).astype(int)
        return df

    def apply_intervention(
        self,
        new_task_df: pd.DataFrame,
        random_seed: RandomState | None = None,
        new_respondents: Literal["error", "population"] = "error",
        draw_batch_size: int | None = None,
    ) -> xr.Dataset:
        """Simulate choices under a counterfactual task design.

        Wraps :meth:`predict_choices` with two conveniences:

        1. Dummy ``is_best`` / ``is_worst`` columns are auto-generated when
           both are absent from ``new_task_df`` (they are unused during
           prediction — only the item layout matters).
        2. The result is stored as ``self.intervention_idata`` for downstream
           comparison, matching the convention in the other customer-choice
           models.

        This is the **Type 1** intervention (observable attribute / assortment
        change). All items in ``new_task_df`` must already be in the trained
        pool. To score *new* items outside the training pool use
        :meth:`score_new_items` (part-worths mode only).

        Parameters
        ----------
        new_task_df : pd.DataFrame
            Counterfactual task design. Must contain ``respondent_id``,
            ``task_id``, and ``item_id`` columns. ``is_best`` / ``is_worst``
            are optional — dummy flags are auto-added when both are absent.
        random_seed : RandomState, optional
            Passed to :meth:`predict_choices`.
        new_respondents : {"error", "population"}, default "error"
            Passed to :meth:`predict_choices`.
        draw_batch_size : int, optional
            Passed to :meth:`predict_choices`.

        Returns
        -------
        xr.Dataset
            Dataset from :meth:`predict_choices` with ``best_pick``,
            ``worst_pick``, ``p_best``, and ``p_worst``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.idata is None:
            raise RuntimeError(
                "apply_intervention requires a fitted model. Call .fit() first."
            )
        new_task_df = self._ensure_best_worst_flags(new_task_df)
        result = self.predict_choices(
            new_task_df,
            random_seed=random_seed,
            new_respondents=new_respondents,
            draw_batch_size=draw_batch_size,
        )
        self.intervention_idata = result
        return result

    def score_new_items(
        self,
        new_item_attributes: pd.DataFrame,
    ) -> xr.Dataset:
        """Compute posterior share-of-preference after introducing new items.

        Part-worths mode only. Each row of ``new_item_attributes`` is a
        hypothetical SKU not seen during training. The fitted patsy formula
        is applied via :meth:`transform_attributes` to obtain design vectors
        for the new items; utilities follow from the posterior of
        ``beta_feat``. The share-of-preference is computed over the *extended*
        pool (training items first, new items appended).

        This is the **Type 2** intervention (market-structure change — adding
        new items to the competitive set).

        Parameters
        ----------
        new_item_attributes : pd.DataFrame
            One row per new item, indexed by a unique item name, with the
            same attribute columns as the training ``item_attributes``. Item
            names must not overlap with the training pool.

        Returns
        -------
        xr.Dataset
            Variables ``u_item`` and ``share_of_preference``, both of shape
            ``(chain, draw, items)``, where the ``items`` coordinate covers
            training items followed by new items. The result is also stored
            as ``self.intervention_idata``.

        Raises
        ------
        RuntimeError
            If called on a non-part-worths model or before fitting.
        ValueError
            If any new item name overlaps with the training pool.
        """
        if not self._is_partworths:
            raise RuntimeError(
                "score_new_items is only available in part-worths mode "
                "(item_attributes + utility_formula must be set at __init__)."
            )
        if self.idata is None:
            raise RuntimeError(
                "score_new_items requires a fitted model. Call .fit() first."
            )
        overlap = set(new_item_attributes.index) & set(self.items)
        if overlap:
            raise ValueError(
                f"new_item_attributes contains item(s) already in the training "
                f"pool: {sorted(overlap)}. Use distinct names for new items."
            )

        X_new = self.transform_attributes(new_item_attributes)  # (N, P)
        if self.X is None:  # pragma: no cover — guarded by _is_partworths
            raise RuntimeError("self.X is unexpectedly None in part-worths mode.")
        X_all = np.vstack([self.X, X_new])  # (I + N, P)
        all_items = list(self.items) + list(new_item_attributes.index)

        posterior = self.idata["posterior"]
        beta_feat_vals = posterior["beta_feat"].values  # (C, D, P)
        # (C, D, I+N) — softmax is location-invariant so centering is unnecessary.
        U_all = np.einsum("cdp,ip->cdi", beta_feat_vals, X_all)
        U_shift = U_all - U_all.max(axis=-1, keepdims=True)
        exp_u = np.exp(U_shift)
        shares = exp_u / exp_u.sum(axis=-1, keepdims=True)

        result = xr.Dataset(
            {
                "u_item": (("chain", "draw", "items"), U_all),
                "share_of_preference": (("chain", "draw", "items"), shares),
            },
            coords={
                "chain": posterior["chain"].values,
                "draw": posterior["draw"].values,
                "items": all_items,
            },
        )
        self.intervention_idata = result
        return result

    @property
    def default_model_config(self) -> dict:
        """Default priors — returns only the priors used by the active mode.

        Part-worths mode uses ``beta_feat`` (and optionally ``sigma_feat``,
        ``z_feat`` for random attributes). Item-intercept mode uses
        ``beta_item_``, and optionally ``sigma_item`` / ``z_item`` when
        ``random_intercepts=True``.
        """
        if self._is_partworths:
            cfg: dict[str, Prior] = {
                "beta_feat": Prior("Normal", mu=0, sigma=1, dims="features"),
            }
            if self.random_attributes:
                cfg["sigma_feat"] = Prior("HalfNormal", sigma=1, dims="random_features")
                cfg["z_feat"] = Prior(
                    "Normal",
                    mu=0,
                    sigma=1,
                    dims=("respondents", "random_features"),
                )
            return cfg
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
        result: dict[str, int | float | dict] = {}
        if self._is_partworths:
            result["beta_feat"] = self.model_config["beta_feat"].to_dict()
            if self.random_attributes:
                result["sigma_feat"] = self.model_config["sigma_feat"].to_dict()
                result["z_feat"] = self.model_config["z_feat"].to_dict()
        else:
            result["beta_item_"] = self.model_config["beta_item_"].to_dict()
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
        if self._is_partworths:
            self.coords["features"] = self.feature_names
            if self.random_attributes:
                self.coords["random_features"] = self.random_attributes
        if self.full_covariance and self.random_intercepts:
            # LKJ Cholesky and correlation matrices are (I, I); PyMC requires
            # two distinct dim names for each axis.
            self.coords["items_bis"] = self.items
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
            Model with ``beta_item``/``beta_item_r`` (intercept mode) or
            ``beta_feat``/``U_item_r`` (part-worths mode) plus two observed
            ``Categorical`` likelihoods ``best_pick`` / ``worst_pick``.
        """
        with pm.Model(coords=self.coords) as model:
            item_idx = pm.Data(
                "item_idx", arrays["item_idx"], dims=("tasks", "positions")
            )
            mask = pm.Data("mask", arrays["mask"], dims=("tasks", "positions"))
            best_pos = pm.Data("best_pos", arrays["best_pos"], dims="tasks")
            worst_pos = pm.Data("worst_pos", arrays["worst_pos"], dims="tasks")
            resp_idx = pm.Data("resp_idx", arrays["resp_idx"], dims="tasks")

            if self._is_partworths:
                U = self._build_partworths_utility(arrays, resp_idx, item_idx)
            else:
                U = self._build_intercept_utility(arrays, resp_idx, item_idx)

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

    def _build_intercept_utility(
        self, arrays: MaxDiffArrays, resp_idx: Any, item_idx: Any
    ) -> Any:
        """Item-intercept utility path (v0.1 behaviour).

        Returns a ``(tasks, positions)`` tensor of per-task per-position
        utilities. The reference item is pinned to 0 for identification;
        respondents may additionally draw item-level deviations when
        ``random_intercepts=True``.
        """
        ref_idx = self.items.index(self.reference_item)

        beta_item_ = self.model_config["beta_item_"].create_variable("beta_item_")
        beta_item = pm.Deterministic(
            "beta_item", pt.set_subtensor(beta_item_[ref_idx], 0.0), dims="items"
        )

        if self.random_intercepts:
            if self.full_covariance:
                # Full LKJ Cholesky covariance: Σ = L L^T where
                # L ~ LKJCholeskyCov(eta, sd_dist=HalfNormal(sigma)).
                # compute_corr=True unpacks to (chol, corr, stds).
                sd_sigma = self.model_config["sigma_item"].parameters.get("sigma", 1.0)
                n_items = len(self.items)
                chol, corr, stds = pm.LKJCholeskyCov(
                    "chol_cov",
                    n=n_items,
                    eta=self.lkj_eta,
                    sd_dist=pm.HalfNormal.dist(sigma=sd_sigma, shape=n_items),
                    compute_corr=True,
                )
                # Store full (I, I) matrices as named Deterministics so they
                # appear in the posterior and are accessible for new-respondent
                # population draws via _draw_new_respondent_utilities.
                chol_L = pm.Deterministic("chol_L", chol, dims=("items", "items_bis"))
                pm.Deterministic("corr_matrix", corr, dims=("items", "items_bis"))
                pm.Deterministic("item_stds", stds, dims="items")

                z_item = self.model_config["z_item"].create_variable("z_item")
                if self.non_centered:
                    # Non-centered reparameterisation: β_r = μ + L z_r
                    # pt.dot(z_item, chol_L.T): (R, I) @ (I, I) → (R, I)
                    beta_item_r = pm.Deterministic(
                        "beta_item_r",
                        beta_item[None, :] + pt.dot(z_item, chol_L.T),
                        dims=("respondents", "items"),
                    )
                else:
                    # Centered: β_r ~ MVNormal(β, chol=L)
                    beta_item_r = pm.MvNormal(
                        "beta_item_r",
                        mu=beta_item,
                        chol=chol_L,
                        dims=("respondents", "items"),
                    )
                return beta_item_r[resp_idx[:, None], item_idx]

            # Diagonal (v0.1/v0.2) path — independent per-item scales.
            sigma_item = self.model_config["sigma_item"].create_variable("sigma_item")
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
            return beta_item_r[resp_idx[:, None], item_idx]

        return beta_item[item_idx]

    def _build_partworths_utility(
        self, arrays: MaxDiffArrays, resp_idx: Any, item_idx: Any
    ) -> Any:
        """Part-worths utility path: ``U_i = X_i^T @ beta_feat``.

        With ``random_attributes`` non-empty, respondents additionally draw
        deviations on the specified feature columns via a non-centered
        parameterisation, giving per-respondent per-item utilities
        ``U_item_r``. Stored as ``pm.Data('X', ...)`` so the design matrix
        can be swapped via ``pm.set_data`` for within-pool predictions.
        """
        n_respondents = len(self.coords["respondents"])

        X_data = pm.Data("X", self.X, dims=("items", "features"))
        beta_feat = self.model_config["beta_feat"].create_variable("beta_feat")

        U_item_pop_raw = pt.dot(X_data, beta_feat)
        U_item_pop = pm.Deterministic(
            "U_item_pop", U_item_pop_raw - pt.mean(U_item_pop_raw), dims="items"
        )

        if not self.random_attributes:
            return U_item_pop[item_idx]

        rc_idx = np.array(
            [self.feature_names.index(rc) for rc in self.random_attributes],
            dtype=np.int64,
        )
        sigma_feat = self.model_config["sigma_feat"].create_variable("sigma_feat")
        z_feat = self.model_config["z_feat"].create_variable("z_feat")

        # Per-respondent deviation on the random-feature subset only.
        # dev has shape (respondents, features) with zeros outside rc columns,
        # so beta_full_r = beta_feat + dev is (respondents, features).
        dev = pt.zeros((n_respondents, len(self.feature_names)))
        dev = pt.set_subtensor(dev[:, rc_idx], z_feat * sigma_feat[None, :])
        beta_full_r = beta_feat[None, :] + dev  # (R, P)
        U_item_r = pm.Deterministic(
            "U_item_r",
            pt.dot(beta_full_r, X_data.T),
            dims=("respondents", "items"),
        )
        U_item_r_centered = U_item_r - pt.mean(U_item_r, axis=1, keepdims=True)
        return U_item_r_centered[resp_idx[:, None], item_idx]

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
        attrs["full_covariance"] = json.dumps(self.full_covariance)
        attrs["lkj_eta"] = json.dumps(self.lkj_eta)
        attrs["utility_formula"] = json.dumps(self.utility_formula)
        attrs["random_attributes"] = json.dumps(self.random_attributes)
        if self._is_partworths and self.item_attributes is not None:
            # Persist item_attributes aligned to self.items so the design
            # matrix can be rebuilt on load via the same patsy formula.
            aligned = self.item_attributes.loc[self.items]
            attrs["item_attributes"] = json.dumps(
                {
                    "columns": list(aligned.columns),
                    "index": list(aligned.index),
                    "index_name": aligned.index.name,
                    "data": aligned.to_dict(orient="list"),
                }
            )
        else:
            attrs["item_attributes"] = json.dumps(None)
        return attrs

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        """Rehydrate init kwargs from serialised idata attrs."""
        item_attributes_raw = json.loads(attrs.get("item_attributes", "null"))
        if item_attributes_raw is None:
            item_attributes = None
        else:
            item_attributes = pd.DataFrame(
                item_attributes_raw["data"], index=item_attributes_raw["index"]
            )[item_attributes_raw["columns"]]
            item_attributes.index.name = item_attributes_raw.get("index_name")
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
            "full_covariance": json.loads(attrs.get("full_covariance", "false")),
            "lkj_eta": json.loads(attrs.get("lkj_eta", "2.0")),
            "utility_formula": json.loads(attrs.get("utility_formula", "null")),
            "random_attributes": json.loads(attrs.get("random_attributes", "[]")),
            "item_attributes": item_attributes,
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

        Appropriate for **in-sample posterior predictive checks** on training
        data. When ``task_df`` is provided the model data containers are
        updated via ``pm.set_data`` before sampling.

        .. warning::

            Due to the sequential best-worst likelihood, ``worst_pick`` is
            conditioned on the *observed* best position, not on the sampled
            ``best_pick``. The joint draw is therefore incoherent for
            generative or counterfactual use. See the class-level
            *Posterior predictive limitations* note for details, and use
            :meth:`predict_choices` / :meth:`apply_intervention` instead for
            any out-of-sample or counterfactual simulation.
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
        new_respondents: Literal["error", "population"] = "error",
        draw_batch_size: int | None = None,
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
            Seed for the numpy Generator used for new-respondent population
            draws and for sampling best / worst. The full output is
            deterministic given this seed, regardless of ``draw_batch_size``.
        new_respondents : {"error", "population"}, default "error"
            How to handle respondents in ``task_df`` that were not in the
            training data, when ``random_intercepts=True``:

            * ``"error"`` (default): raise ``ValueError``.
            * ``"population"``: for each unknown respondent, draw a fresh
              respondent-level utility vector from the fitted population
              distribution ``Normal(beta_item, sigma_item)`` per posterior
              sample. This is the standard mixed-logit extrapolation to a
              brand-new customer.

            Ignored when ``random_intercepts=False`` (no respondent-level
            parameters exist).
        draw_batch_size : int, optional
            If provided, compute the per-task utilities in chunks of this many
            posterior draws rather than materializing the full
            ``(chain, draw, tasks, positions)`` tensor at once. Lowers peak
            memory linearly at the cost of a small Python-level loop. Output
            is bit-identical to the unbatched path for a given ``random_seed``.

        Returns
        -------
        xr.Dataset
            ``posterior_predictive``-shaped dataset with ``best_pick`` and
            ``worst_pick`` variables of shape ``(chain, draw, tasks)`` and
            ``p_best`` / ``p_worst`` of shape ``(chain, draw, tasks, positions)``.
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
        rng = (
            random_seed
            if isinstance(random_seed, np.random.Generator)
            else np.random.default_rng(random_seed)
        )

        # Resolve the per-respondent or population beta tensor aligned with
        # ``arrays["respondent_to_idx"]`` so downstream indexing via
        # ``resp_idx`` is direct. Population draws for unknown respondents
        # happen once, up-front, so batched output matches unbatched exactly.
        pred_beta_r, beta_pop = self._resolve_predict_beta(
            posterior, arrays, new_respondents, rng
        )

        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        T = arrays["n_tasks"]
        k_max = arrays["k_max"]

        # Pre-draw all categorical uniforms so batched and unbatched paths
        # produce identical output given the same seed.
        u_best_full = rng.random((n_chains, n_draws, T))
        u_worst_full = rng.random((n_chains, n_draws, T))

        if draw_batch_size is None or draw_batch_size >= n_draws:
            batch_slices: list[slice] = [slice(0, n_draws)]
        else:
            batch_slices = [
                slice(s, min(s + draw_batch_size, n_draws))
                for s in range(0, n_draws, draw_batch_size)
            ]

        best_all = np.empty((n_chains, n_draws, T), dtype=np.int64)
        worst_all = np.empty((n_chains, n_draws, T), dtype=np.int64)
        p_best_all = np.empty((n_chains, n_draws, T, k_max), dtype=np.float64)
        p_worst_all = np.empty((n_chains, n_draws, T, k_max), dtype=np.float64)

        for sl in batch_slices:
            if pred_beta_r is not None:
                beta_slice = pred_beta_r[:, sl, :, :]
            elif beta_pop is not None:
                beta_slice = beta_pop[:, sl, :]
            else:  # pragma: no cover - guarded by _resolve_predict_beta
                raise RuntimeError(
                    "_resolve_predict_beta must return exactly one non-None tensor."
                )
            bb, wb, pb, pw = self._predict_choices_batch(
                beta_slice,
                arrays,
                u_best_full[:, sl, :],
                u_worst_full[:, sl, :],
            )
            best_all[:, sl] = bb
            worst_all[:, sl] = wb
            p_best_all[:, sl] = pb
            p_worst_all[:, sl] = pw

        return xr.Dataset(
            {
                "best_pick": (("chain", "draw", "tasks"), best_all),
                "worst_pick": (("chain", "draw", "tasks"), worst_all),
                "p_best": (("chain", "draw", "tasks", "positions"), p_best_all),
                "p_worst": (("chain", "draw", "tasks", "positions"), p_worst_all),
            },
            coords={
                "chain": posterior["chain"].values,
                "draw": posterior["draw"].values,
                "tasks": np.arange(T),
                "positions": np.arange(k_max),
            },
        )

    @property
    def _has_respondent_variation(self) -> bool:
        """True iff the model has per-respondent utility parameters."""
        if self._is_partworths:
            return bool(self.random_attributes)
        return self.random_intercepts

    def _resolve_predict_beta(
        self,
        posterior: xr.Dataset,
        arrays: MaxDiffArrays,
        new_respondents: Literal["error", "population"],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Assemble the per-prediction utility tensor aligned with pred respondents.

        Returns ``(pred_beta_r, None)`` of shape ``(chain, draw, n_pred, n_items)``
        when the model has per-respondent variation, or ``(None, beta_pop)`` of
        shape ``(chain, draw, n_items)`` otherwise. Works for both item-intercept
        and part-worths modes; unknown respondents are either rejected or drawn
        from the fitted population distribution.
        """
        # Population utilities used both as the fall-through in
        # random-intercept-free models and as the mean of the population
        # distribution when drawing utilities for new respondents.
        if self._is_partworths:
            pop_key = "U_item_pop"
            per_resp_key = "U_item_r"
        else:
            pop_key = "beta_item"
            per_resp_key = "beta_item_r"

        if not self._has_respondent_variation:
            return None, posterior[pop_key].values

        training_respondents = list(self.coords["respondents"])
        resp_name_to_training_idx = {r: i for i, r in enumerate(training_respondents)}

        pred_respondents = list(arrays["respondent_to_idx"].keys())
        unknown_mask = np.array(
            [r not in resp_name_to_training_idx for r in pred_respondents]
        )
        n_unknown = int(unknown_mask.sum())

        if n_unknown > 0 and new_respondents == "error":
            example = pred_respondents[int(np.argmax(unknown_mask))]
            raise ValueError(
                f"{n_unknown} respondent(s) in task_df were not in the "
                f"training data (e.g. {example!r}). Pass "
                "new_respondents='population' to draw their utilities from "
                "the fitted population distribution, or restrict task_df to "
                "training respondents."
            )

        beta_r_train = posterior[per_resp_key].values  # (C, D, R_train, I)
        n_chains, n_draws, _, n_items = beta_r_train.shape
        n_pred = len(pred_respondents)

        pred_beta_r = np.empty(
            (n_chains, n_draws, n_pred, n_items), dtype=beta_r_train.dtype
        )

        known_positions = np.where(~unknown_mask)[0]
        if known_positions.size > 0:
            known_train_idx = np.array(
                [
                    resp_name_to_training_idx[pred_respondents[p]]
                    for p in known_positions
                ],
                dtype=np.int64,
            )
            pred_beta_r[:, :, known_positions, :] = beta_r_train[
                :, :, known_train_idx, :
            ]

        if n_unknown > 0:
            unknown_positions = np.where(unknown_mask)[0]
            pred_beta_r[:, :, unknown_positions, :] = (
                self._draw_new_respondent_utilities(posterior, n_unknown, rng)
            )

        return pred_beta_r, None

    def _draw_new_respondent_utilities(
        self,
        posterior: xr.Dataset,
        n_new: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw fresh per-item utility vectors for ``n_new`` new respondents.

        In intercept mode the draws come from ``Normal(beta_item, sigma_item)``.
        In part-worths mode we draw fresh respondent-specific coefficients from
        ``Normal(beta_feat_rc, sigma_feat)`` on the random-feature subset, hold
        fixed features at their population value, then map to per-item utility
        via the stored design matrix ``self.X``.

        Returns an array of shape ``(chain, draw, n_new, n_items)``.
        """
        if self._is_partworths:
            beta_feat = posterior["beta_feat"].values  # (C, D, P)
            sigma_feat = posterior["sigma_feat"].values  # (C, D, P_rc)
            n_chains, n_draws, n_features = beta_feat.shape
            n_items = len(self.items)
            rc_idx = np.array(
                [self.feature_names.index(rc) for rc in self.random_attributes],
                dtype=np.int64,
            )
            eps = rng.standard_normal((n_chains, n_draws, n_new, len(rc_idx)))
            beta_full = np.broadcast_to(
                beta_feat[:, :, None, :],
                (n_chains, n_draws, n_new, n_features),
            ).copy()
            beta_full[:, :, :, rc_idx] += eps * sigma_feat[:, :, None, :]
            # (C, D, n_new, P) @ (P, I) -> (C, D, n_new, I).
            # Part-worths mode guarantees self.X was populated by
            # _prepare_design_matrix during __init__.
            if self.X is None:  # pragma: no cover - guarded by _is_partworths
                raise RuntimeError("self.X is unexpectedly None in part-worths mode.")
            return beta_full @ self.X.T

        beta_item_post = posterior["beta_item"].values  # (C, D, I)
        n_chains, n_draws, n_items = beta_item_post.shape

        if self.full_covariance:
            # chol_L stored as Deterministic (C, D, I, I) lower-triangular.
            chol_L_post = posterior["chol_L"].values  # (C, D, I, I)
            z = rng.standard_normal((n_chains, n_draws, n_new, n_items))
            # Vectorised MVN draw: β_r = β + L z_r
            # "cdij,cdrj->cdri": for each (c,d), L @ z[r,:] for every new r.
            return beta_item_post[:, :, None, :] + np.einsum(
                "cdij,cdrj->cdri", chol_L_post, z
            )

        sigma_item_post = posterior["sigma_item"].values  # (C, D, I)
        eps = rng.standard_normal((n_chains, n_draws, n_new, n_items))
        return beta_item_post[:, :, None, :] + eps * sigma_item_post[:, :, None, :]

    def _predict_choices_batch(
        self,
        beta_slice: np.ndarray,
        arrays: MaxDiffArrays,
        u_best: np.ndarray,
        u_worst: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute (best, worst, p_best, p_worst) for one slice of posterior draws.

        ``beta_slice`` is either ``(C, D_b, n_pred, I)`` (random intercepts) or
        ``(C, D_b, I)`` (population only). ``u_best`` / ``u_worst`` are
        pre-drawn uniforms of shape ``(C, D_b, T)``. Returns arrays of shape
        ``(C, D_b, T)`` for samples and ``(C, D_b, T, K_max)`` for probabilities.
        """
        item_idx = arrays["item_idx"]  # (T, K_max)
        resp_idx = arrays["resp_idx"]  # (T,)
        mask = arrays["mask"]  # (T, K_max)
        T = item_idx.shape[0]

        if beta_slice.ndim == 4:
            # Fancy-index directly to (C, D_b, T, K_max) — avoids the larger
            # (C, D_b, T, I) intermediate that the v0.1 path materialized.
            U = beta_slice[:, :, resp_idx[:, None], item_idx]
        else:
            U = beta_slice[:, :, item_idx]  # (C, D_b, T, K_max)

        mask_bcast = mask[None, None, :, :]
        U_best_masked = np.where(mask_bcast, U, NEG_INF)
        p_best = _softmax_stable(U_best_masked, axis=-1)
        best_samples = _sample_categorical_from_u(p_best, u_best)

        chain, draw_b = U.shape[:2]
        U_worst_signflip = np.where(mask_bcast, -U, NEG_INF)
        chain_idx = np.arange(chain)[:, None, None]
        draw_idx = np.arange(draw_b)[None, :, None]
        flat_tasks = np.broadcast_to(np.arange(T)[None, None, :], (chain, draw_b, T))
        U_worst_masked = U_worst_signflip.copy()
        U_worst_masked[chain_idx, draw_idx, flat_tasks, best_samples] = NEG_INF
        p_worst = _softmax_stable(U_worst_masked, axis=-1)
        worst_samples = _sample_categorical_from_u(p_worst, u_worst)

        return best_samples, worst_samples, p_best, p_worst

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
