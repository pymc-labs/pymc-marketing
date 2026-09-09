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
r"""Predicted Incrementality by Experimentation (PIE) model.

Randomised controlled trials (RCTs) — geo experiments or ghost-ad holdouts —
are considered the gold standard for measuring the *incremental* effect of an ad
campaign. However, they are costly and slow, so advertisers only typically run them
for a fraction of their campaigns. PIE turns that fraction into leverage by fitting
a supervised model on the corpus of campaigns that *did* receive an RCT,
learning the map from observable campaign features to experimentally measured
incrementality, then predicts incrementality for the campaigns that never ran
an experiment.

For campaign :math:`i` with feature vector :math:`x_i` and RCT-measured
incrementality :math:`\tau_i`, PIE models

.. math::

    \tau_i = f(x_i) + \varepsilon_i, \quad
    \varepsilon_i \sim \mathrm{Normal}(0, \sigma),

where :math:`f` is a Bayesian Additive Regression Trees (BART) ensemble — a
sum of regularised regression trees. Because :math:`f` is sampled rather than
point-estimated, the predicted incrementality for a new campaign with features
:math:`x_\star` is a full posterior over :math:`f(x_\star)`.

The mean function is expressed with the **terms** framework
(:mod:`pymc_marketing.terms`): by default a
:class:`~pymc_marketing.bart.Bart` term wrapping ``pymc_bart`` built from the
``"bart"`` configuration keys, but the mean function is configuration, not
code — ``model_config['mu']`` takes any term recipe
(:class:`~pymc_marketing.bart.Bart`, :class:`~pymc_marketing.terms.Dot`,
compositions thereof), so the ensemble is swapped for a plain linear
regression by passing a ``Dot`` recipe. See the :class:`PIEModel` docstring
for details.

The approach rests on three assumptions:

1. the RCT corpus is representative of the campaigns being predicted (predictions
far outside the corpus's feature support are extrapolation and unreliable)
2. the recorded features carry enough signal to explain variation in incrementality
3. measured incrementality is a consistent estimate of the true causal effect
(per-RCT measurement error is not yet modelled — see :class:`PIEModel`).

For the full method, see [1]_.

References
----------
.. [1] Gordon, B. R., Moakler, R., & Zettelmeyer, F. (2026).
   Predicted Incrementality by Experimentation (PIE) for Ad Measurement.
   NBER Working Paper No. 35044.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pydantic import Field, validate_call
from pymc_extras.prior import Prior
from sklearn.preprocessing import LabelEncoder

from pymc_marketing.model_builder import RegressionModelBuilder
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.serialization import serialization
from pymc_marketing.terms import (
    ModelTerm,
    build_param,
    collect_coords,
    collect_terms,
    register_data,
)
from pymc_marketing.terms import set_data as set_term_data


def _is_categorical(series: pd.Series) -> bool:
    """Return True if the series should be label-encoded.

    Any non-numeric dtype is treated as categorical. This covers ``object``,
    pandas ``CategoricalDtype``, ``StringDtype`` (incl. ``string[pyarrow]``),
    ``pd.ArrowDtype(pa.string())``, and similar extension dtypes — so callers
    can pass DataFrames produced by ``df.convert_dtypes()`` or a pyarrow CSV
    backend without first casting to ``object``/``category``.
    """
    return not pd.api.types.is_numeric_dtype(series)


def make_split_rules(
    categorical_split: str,
    column_names: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> list[Any]:
    """Build BART split rules matching the training column order.

    Parameters
    ----------
    categorical_split : str
        Split rule mode: ``"onehot"`` assigns
        :class:`~pymc_bart.split_rules.OneHotSplitRule` to label-encoded
        categorical columns so splits are "level X vs not-X" rather than
        "encoded value < c", or ``"continuous"`` to use ordered splits
        everywhere.
    column_names : list[str]
        Feature columns, in the order they are fed to BART.
    encoders : dict, optional
        Label encoders keyed by column name; the keys determine which
        columns are categorical in the ``"onehot"`` mode.

    Returns
    -------
    list
        One split rule per column, in order.

    Raises
    ------
    ValueError
        If ``categorical_split`` is not ``"onehot"`` or ``"continuous"``.
    """
    from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule

    if categorical_split not in ("onehot", "continuous"):
        raise ValueError(
            f"categorical_split must be 'onehot' or 'continuous', "
            f"got {categorical_split!r}."
        )
    if categorical_split == "onehot":
        encoders = encoders or {}
        return [
            OneHotSplitRule() if col in encoders else ContinuousSplitRule()
            for col in column_names
        ]
    return [ContinuousSplitRule() for _ in column_names]


class PIEModel(RegressionModelBuilder):
    """Predicted Incrementality by Experimentation model.

    Trains a Bayesian BART regression on a corpus of past RCTs mapping
    campaign features to measured incrementality, then predicts incrementality
    for non-experimental campaigns.

    Parameters
    ----------
    pre_determined_features : list[str]
        Feature columns known before the campaign runs (e.g. objective,
        vertical, budget, audience_type). **In the current alpha
        implementation this list is concatenated with**
        ``post_determined_features`` **and fed identically into BART**; the
        distinction is recorded for future versions that gate prediction on
        feature availability but has no effect on the model graph today.
    post_determined_features : list[str]
        Feature columns known only after the campaign runs (e.g.
        exposure_rate, ctr, last_click_conversions_per_dollar,
        avg_treated_outcome). See note above — treated identically to
        ``pre_determined_features`` in this release.
    target_column : str
        Name used for the target variable in the PyMC graph and in saved
        idata groups (``posterior_predictive[target_column]``,
        ``fit_data[target_column]``). Does not select a column from X — X
        and y are always passed separately. Defaults to ``"y"``.
    model_config : dict, optional
        Override default priors / BART settings. Top-level keys merge with
        :py:meth:`default_model_config`; nested dicts (e.g. ``"bart"``) are
        replaced wholesale, so a partial ``"bart"`` override must restate
        every required key (``m``, ``alpha``, ``beta``). Keys:

        - ``"bart"``: dict with ``m`` (int), ``alpha`` (float), ``beta``
          (float), and optional ``response`` — ``"constant"`` (default,
          piecewise-constant leaves), ``"linear"``, or ``"mix"`` (the latter
          two fit linear models in the leaves, which can help on smooth
          response surfaces).
        - ``"sigma"``: :class:`pymc_extras.prior.Prior` for the noise std.
        - ``"categorical_split"``: ``"onehot"`` (default) or ``"continuous"``.
          Controls how label-encoded categorical columns are split by BART
          — see Notes.
        - ``"mu"``: a :class:`~pymc_marketing.terms.ModelTerm` recipe (or its
          serialized dict) for the likelihood mean, taking precedence over the
          default BART recipe built from ``"bart"``. Terms configure their own
          hyperparameters, so the model is a plain linear regression by
          passing e.g.
          ``Dot(var_name="X", name="mu_coef", prior=Prior("Normal", dims="feature"))``
          — which removes the ``pymc-bart`` requirement — or a custom
          :class:`~pymc_marketing.bart.Bart` instance. See
          :mod:`pymc_marketing.bart` for the swappable BART term.
    sampler_config : dict, optional
        Passed to :func:`pymc.sample`. Defaults to ``{}``.

    Examples
    --------
    .. code-block:: python

        import pandas as pd

        from pymc_marketing.pie import PIEModel

        # Corpus of past campaigns, each labelled with the incrementality
        # measured by its RCT.
        X = pd.DataFrame(
            {
                "objective": ["conversions", "traffic", "awareness", "traffic"],
                "vertical": ["retail", "travel", "finance", "retail"],
                "budget": [50_000, 12_000, 80_000, 30_000],
                "exposure_rate": [0.42, 0.71, 0.33, 0.55],
            }
        )
        y = pd.Series([0.81, 0.34, 1.12, 0.49])  # incrementality per dollar

        model = PIEModel(
            pre_determined_features=["objective", "vertical", "budget"],
            post_determined_features=["exposure_rate"],
        )
        model.fit(X, y, random_seed=42)
        preds = model.sample_posterior_predictive(X)

    Notes
    -----
    **This module is alpha — the API and defaults may change.** Tracked
    deviations from the paper [1]_:

    - The paper uses a random forest fit to 2,226 RCTs; this implementation
      uses Bayesian Additive Regression Trees (PyMC-BART) for native
      posterior uncertainty.
    - The paper's decision-theoretic framework (Type I/II error rates,
      disagreement vs RCT-based go/no-go decisions; paper §6) is not
      implemented.
    - Within-campaign sample splitting (paper §4.2) — which breaks the
      mechanical correlation between post-determined features and the
      target — is not implemented.
    - Extrapolation / cold-start diagnostics across advertiser segments
      (paper §5.3) are not implemented.
    - The footnote-2 measurement-error layer
      ``y_observed ~ Normal(y_true, se_rct)`` for per-RCT standard errors is
      not implemented.

    Categorical columns (``object`` or ``category`` dtype) are label-encoded
    in ``build_model``. With ``categorical_split="onehot"`` (default), BART
    uses :class:`pymc_bart.split_rules.OneHotSplitRule` for those columns so
    that splits are "level X vs not-X" rather than "encoded value < c" — this
    avoids imposing the encoder's alphabetical ordering on unordered
    categories. Set ``categorical_split="continuous"`` to fall back to
    ordered splits.

    References
    ----------
    .. [1] Gordon, B. R., Moakler, R., & Zettelmeyer, F. (2026).
       Predicted Incrementality by Experimentation (PIE) for Ad Measurement.
       NBER Working Paper No. 35044.
    """

    _model_type = "PIE Model"
    version = "0.2.0"

    @property
    def output_var(self) -> str:
        """Name of the target variable in the PyMC graph and saved idata."""
        return self.target_column

    @validate_call
    def __init__(
        self,
        *,
        pre_determined_features: list[str] = Field(
            ..., description="Feature columns known before the campaign runs."
        ),
        post_determined_features: list[str] = Field(
            ..., description="Feature columns known only after the campaign runs."
        ),
        target_column: str = Field(
            "y", description="Label for the target variable in idata."
        ),
        model_config: dict | None = Field(None),
        sampler_config: dict | None = Field(None),
    ) -> None:
        super().__init__(model_config=model_config, sampler_config=sampler_config)

        # Parse after merging with defaults, matching CLVModel. This is what
        # rejects legacy dict-format priors with a migration hint; without it
        # they fail much later with an opaque AttributeError.
        self.model_config = parse_model_config(self.model_config)

        self.pre_determined_features = list(pre_determined_features)
        self.post_determined_features = list(post_determined_features)
        self.target_column = target_column
        self._encoders: dict[str, LabelEncoder] = {}
        self._feature_columns: list[str] = []
        self._target_scale: float = 1.0
        self._mu_recipe: ModelTerm | None = None
        self._recipe_sample_vars: list[str] = []

    @property
    def default_model_config(self) -> dict:
        """Default BART hyperparameters, noise prior, and categorical split mode.

        Returns
        -------
        dict
            Keys: ``"bart"`` (dict of BART settings), ``"sigma"`` (Prior),
            ``"categorical_split"`` (str — ``"onehot"`` or ``"continuous"``).
        """
        return {
            "bart": {"m": 200, "alpha": 0.95, "beta": 2.0, "response": "constant"},
            "sigma": Prior("HalfNormal", sigma=1.0),
            "categorical_split": "onehot",
        }

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration (empty — PyMC auto-assigns PGBART + NUTS)."""
        return {}

    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        # Build a fresh dict (not a view of ``self.model_config``) so a
        # downstream mutation of the result cannot leak back into the model,
        # and tolerate a partial override that dropped a top-level key.
        keys = ("bart", "sigma", "categorical_split", "mu")
        return {key: self.model_config[key] for key in keys if key in self.model_config}

    def _make_split_rules(self, column_names: list[str]) -> list[Any]:
        """Build the split rules from the deprecated method entry point.

        .. deprecated:: 0.2.0
            Use :func:`pymc_marketing.pie.model.make_split_rules` instead.
        """
        warnings.warn(
            "PIEModel._make_split_rules is deprecated; use "
            "pymc_marketing.pie.model.make_split_rules instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return make_split_rules(
            self.model_config.get("categorical_split", "onehot"),
            column_names,
            self._encoders,
        )

    def _default_bart_recipe(self) -> ModelTerm:
        """Build the default BART recipe from the ``"bart"`` config keys."""
        from pymc_marketing import bart as bart_module

        if bart_module.pmb is None:
            raise ImportError(
                "pymc-bart is required for PIEModel. "
                "Install it with: pip install 'pymc-marketing[pie]'"
            )

        bart_cfg = self.model_config.get("bart", {})
        missing_bart_keys = {"m", "alpha", "beta"} - set(bart_cfg)
        if missing_bart_keys:
            raise ValueError(
                f"model_config['bart'] is missing required keys: "
                f"{sorted(missing_bart_keys)}. Nested config dicts are replaced "
                "wholesale rather than deep-merged, so a partial 'bart' override "
                "must restate every required key (m, alpha, beta)."
            )
        response = bart_cfg.get("response", "constant")
        if response not in ("constant", "linear", "mix"):
            raise ValueError(
                "model_config['bart']['response'] must be 'constant', 'linear', "
                f"or 'mix', got {response!r}."
            )

        from pymc_marketing.bart import Bart

        return Bart(
            var_name="X",
            y_name="y_obs",
            m=bart_cfg["m"],
            alpha=bart_cfg["alpha"],
            beta=bart_cfg["beta"],
            response=response,
            split_rules=make_split_rules(
                self.model_config.get("categorical_split", "onehot"),
                self._feature_columns,
                self._encoders,
            ),
            name="bart",
        )

    def _resolve_mu_recipe(self) -> ModelTerm:
        """Resolve the likelihood mean recipe from the model configuration.

        ``model_config["mu"]`` (a live term or serialized dict) takes
        precedence; otherwise a :class:`~pymc_marketing.bart.Bart` recipe is
        built from the ``"bart"`` configuration keys.

        Returns
        -------
        ModelTerm
            Recipe for the likelihood mean over the observation dimension.
        """
        config = self.model_config
        if "mu" in config:
            recipe = config["mu"]
            if isinstance(recipe, ModelTerm):
                return recipe
            if isinstance(recipe, dict) and "__type__" in recipe:
                from pymc_marketing import bart as bart_module

                if bart_module.pmb is None:
                    raise ImportError(
                        "pymc-bart is required to rebuild the serialized 'mu' "
                        "recipe of PIEModel. Install it with: "
                        "pip install 'pymc-marketing[pie]'"
                    )
                return serialization.deserialize(recipe)
            raise ValueError(
                "model_config['mu'] must be a ModelTerm or a serialized term "
                "dict with a '__type__' key, got "
                f"{type(recipe).__name__}."
            )
        return self._default_bart_recipe()

    def build_model(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Build the PyMC model graph.

        Parameters
        ----------
        X : pd.DataFrame
            Campaign feature matrix. Only the ``pre_determined_features`` and
            ``post_determined_features`` columns are used — any other columns
            (e.g. an id column) are ignored. Columns with non-numeric dtype
            are label-encoded automatically.
        y : pd.Series or np.ndarray
            Measured incrementality (one value per campaign).
        """
        feature_cols = self.pre_determined_features + self.post_determined_features
        missing = set(feature_cols) - set(X.columns)
        if missing:
            raise ValueError(
                f"Features not found in X: {sorted(missing)}. "
                f"Available columns: {sorted(X.columns)}"
            )

        # Train only on the declared pre/post features, in a deterministic
        # order. Any other columns in X are dropped here so they are never
        # label-encoded or fed to the mean-function recipe.
        self._feature_columns = feature_cols
        X = X.loc[:, feature_cols]
        X_encoded = X.copy()
        self._encoders = {}
        for col in X.columns:
            if _is_categorical(X[col]):
                enc = LabelEncoder()
                X_encoded[col] = enc.fit_transform(X[col].astype(str))
                self._encoders[col] = enc

        y_vals = (
            y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
        )
        if len(y_vals) != len(X):
            raise ValueError(
                f"X and y must have the same length. Got {len(X)} and {len(y_vals)}."
            )
        if not np.all(np.isfinite(y_vals)):
            raise ValueError("y must contain only finite values.")
        abs_max = float(np.max(np.abs(y_vals)))
        self._target_scale = abs_max if abs_max > 0.0 else 1.0
        y_scaled = (y_vals / self._target_scale).astype(float)

        recipe = self._resolve_mu_recipe()
        self._mu_recipe = recipe
        self._recipe_sample_vars = [
            sample_var
            for term in collect_terms([recipe])
            for sample_var in getattr(term, "sample_vars", [])
        ]

        ds = xr.Dataset(
            {
                "X": (("obs", "feature"), X_encoded.to_numpy(dtype=float)),
                "y_obs": (("obs",), y_scaled),
            },
            coords={
                "obs": X_encoded.index.tolist(),
                "feature": X_encoded.columns.tolist(),
            },
        )
        coords = collect_coords(recipe, ds=ds)
        coords["obs"] = ds.coords["obs"].values.tolist()

        with pm.Model(coords=coords) as self.model:
            register_data(recipe, ds=ds)
            model = pm.modelcontext(None)
            if "y_obs" not in model:
                pmd.Data("y_obs", ds["y_obs"])
            mu_det = pmd.Deterministic("mu", build_param(recipe), dims="obs")
            # Plain (non-dims) free variables: PGBART's logp machinery can
            # only replace plain-tensor values, so any xtensor free variable
            # breaks BART sampling.
            sigma = self.model_config["sigma"].create_variable("sigma")
            pmd.Normal(
                self.output_var,
                mu=mu_det,
                sigma=sigma,
                observed=model["y_obs"],
                dims="obs",
            )

    def _data_setter(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        """Swap data inside the PyMC model for out-of-sample prediction.

        Parameters
        ----------
        X : pd.DataFrame
            New feature matrix. Must contain the training feature columns;
            any extra columns are ignored (consistent with ``build_model``).
        y : pd.Series or np.ndarray, optional
            Target values. When ``None`` (predict mode), dummy zeros are used.
        """
        missing = set(self._feature_columns) - set(X.columns)
        if missing:
            raise ValueError(
                f"X is missing required feature columns: {sorted(missing)}."
            )

        X_encoded = X.loc[:, self._feature_columns].copy()
        for col, enc in self._encoders.items():
            values = X_encoded[col].astype(str)
            unseen = set(values) - set(enc.classes_)
            if unseen:
                raise ValueError(
                    f"Column '{col}' contains unseen categories: {sorted(unseen)}. "
                    f"Known categories: {sorted(enc.classes_)}"
                )
            X_encoded[col] = enc.transform(values)

        if y is not None:
            y_vals = (
                y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
            )
            if len(y_vals) != len(X_encoded):
                raise ValueError(
                    f"X and y must have the same length. Got {len(X_encoded)} and {len(y_vals)}."
                )
            if not np.all(np.isfinite(y_vals)):
                raise ValueError("y must contain only finite values.")
            y_data = y_vals / self._target_scale
        else:
            y_data = np.zeros(len(X_encoded), dtype=float)

        if self._mu_recipe is None:
            raise RuntimeError(
                "The model has no mean-function recipe; call fit() or "
                "build_model() before setting prediction data."
            )
        ds_pred = xr.Dataset(
            {"X": (("obs", "feature"), X_encoded.to_numpy(dtype=float))},
            coords={
                "obs": X_encoded.index.tolist(),
                "feature": self._feature_columns,
            },
        )
        with self.model:
            set_term_data(self._mu_recipe, ds=ds_pred, model=self.model)
            pm.set_data({"y_obs": y_data}, coords={"obs": X_encoded.index.tolist()})

    def sample_posterior_predictive(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Sample posterior predictive draws and return in the original target scale.

        Parameters
        ----------
        X : pd.DataFrame
            Campaign feature matrix for prediction. Same schema as training X
            (no target column required).
        extend_idata : bool
            Whether to attach predictions to ``self.idata``. Defaults to True.
        combined : bool
            Whether to combine chain and draw dims into a single sample dim.
            Defaults to True.

        Returns
        -------
        xarray.Dataset
            Posterior predictive draws in the original target scale, containing
            ``self.output_var``. The output variable has shape ``(obs, sample)``
            when ``combined=True``.
        """
        self._data_setter(X)

        sample_vars = list(
            dict.fromkeys([*self._recipe_sample_vars, "mu", self.output_var])
        )
        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata,
                sample_vars=sample_vars,
                **kwargs,
            )

        variable_name = (
            "predictions" if kwargs.get("predictions") else "posterior_predictive"
        )
        group = post_pred[variable_name]
        if self.output_var in group:
            group[self.output_var] = group[self.output_var] * self._target_scale

        if extend_idata:
            self.idata.update(post_pred)  # type: ignore[union-attr]

        return az.extract(
            post_pred,
            variable_name,
            combined=combined,
            keep_dataset=True,
        )

    def predict_posterior(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Posterior predictive draws for ``X`` as a single DataArray.

        Overrides :meth:`RegressionModelBuilder.predict_posterior` to skip the
        inherited :func:`sklearn.utils.check_array` validation, which would
        coerce the feature DataFrame to a numeric ndarray (dropping column
        names and rejecting the categorical columns ``PIEModel`` label-encodes
        itself).

        Parameters
        ----------
        X : pd.DataFrame
            Campaign feature matrix for prediction. Same schema as training X.
        extend_idata : bool
            Whether to attach predictions to ``self.idata``. Defaults to True.
        combined : bool
            Whether to combine chain and draw dims into a single sample dim.
            Defaults to True.

        Returns
        -------
        xarray.DataArray
            Posterior predictive draws for ``self.output_var`` in the original
            target scale.
        """
        posterior_predictive_samples = self.sample_posterior_predictive(
            X, extend_idata=extend_idata, combined=combined, **kwargs
        )
        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior "
                "predictive samples."
            )
        return posterior_predictive_samples[self.output_var]

    def build_from_idata(self, idata: xr.DataTree) -> None:
        """Rebuild the model from saved inference data.

        Calls the inherited :meth:`RegressionModelBuilder.build_from_idata`
        (which re-runs :meth:`build_model` on the saved ``fit_data`` and
        therefore recomputes ``_target_scale``), then overwrites
        ``_target_scale`` with the value stored in ``idata.attrs`` to
        preserve the exact scale used at fit time (avoids round-off if
        ``fit_data`` was edited or re-serialised).
        """
        super().build_from_idata(idata)
        if "target_scale" in idata.attrs:
            self._target_scale = float(idata.attrs["target_scale"])

    def create_idata_attrs(self) -> dict[str, str]:
        """Extend the base idata attrs with PIEModel-specific fields.

        Adds ``target_column``, ``pre_determined_features``,
        ``post_determined_features``, and ``target_scale``.
        """
        attrs = super().create_idata_attrs()
        attrs["target_column"] = self.target_column
        attrs["pre_determined_features"] = json.dumps(self.pre_determined_features)
        attrs["post_determined_features"] = json.dumps(self.post_determined_features)
        attrs["target_scale"] = str(self._target_scale)
        return attrs

    @classmethod
    def attrs_to_init_kwargs(cls, attrs: dict) -> dict[str, Any]:
        """Reconstruct constructor kwargs from saved idata attrs."""
        kwargs = super().attrs_to_init_kwargs(attrs)
        kwargs["target_column"] = attrs["target_column"]
        kwargs["pre_determined_features"] = json.loads(attrs["pre_determined_features"])
        kwargs["post_determined_features"] = json.loads(
            attrs["post_determined_features"]
        )
        # `kwargs.get("model_config", {})` is not enough: a serialised
        # `model_config` of `null` yields None, and `None.get(...)` raises.
        model_config = kwargs.get("model_config") or {}
        if isinstance(model_config.get("sigma"), dict):
            model_config["sigma"] = Prior.from_dict(model_config["sigma"])
        return kwargs
