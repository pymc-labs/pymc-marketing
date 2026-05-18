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
"""Predicted Incrementality by Experimentation (PIE) model."""

from __future__ import annotations

import json
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import Field, validate_call
from pymc_extras.prior import Prior
from sklearn.preprocessing import LabelEncoder

from pymc_marketing.model_builder import RegressionModelBuilder

try:
    import pymc_bart as pmb
    from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule
except ImportError:
    pmb = None  # type: ignore[assignment]
    ContinuousSplitRule = None  # type: ignore[assignment,misc]
    OneHotSplitRule = None  # type: ignore[assignment,misc]


def generate_synthetic_rct_corpus(
    n_campaigns: int = 500,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic RCT corpus for PIEModel demonstration and testing.

    Parameters
    ----------
    n_campaigns : int
        Number of campaign rows to generate. Default 500.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: campaign_id, objective, vertical, audience_type
        (all object dtype), budget, exposure_rate, ctr, avg_treated_outcome,
        last_click_conversions_per_dollar, measured_incrementality_per_dollar.

    Notes
    -----
    The DGP is a linear model over one-hot-expanded categorical features and
    scaled continuous features. ``last_click_conversions_per_dollar`` is a
    biased-but-informative proxy for the true incrementality signal, matching
    the paper's qualitative framing better than independent noise.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm import generate_synthetic_rct_corpus

        df = generate_synthetic_rct_corpus(n_campaigns=200, seed=42)

    References
    ----------
    .. [1] Gordon, B. R., Moakler, R., & Zettelmeyer, F. (2026).
       Predicted Incrementality by Experimentation (PIE) for Ad Measurement.
       NBER Working Paper No. 35044.
    """
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

    # One-hot encode categoricals for DGP only (not passed to PIEModel)
    obj_oh = (objective[:, None] == objectives).astype(float)  # (n, 3)
    vert_oh = (vertical[:, None] == verticals).astype(float)  # (n, 3)
    aud_oh = (audience_type[:, None] == audience_types).astype(float)  # (n, 2)

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
        [
            0.30,
            -0.10,
            0.20,  # objective
            0.40,
            -0.20,
            0.10,  # vertical
            0.50,
            -0.30,  # audience_type
            0.20,  # budget (normalised)
            0.60,  # exposure_rate
            -0.10,  # ctr
            0.30,  # avg_treated_outcome (normalised)
        ]
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


def _is_categorical(series: pd.Series) -> bool:
    """Return True if the series should be label-encoded.

    Any non-numeric dtype is treated as categorical. This covers ``object``,
    pandas ``CategoricalDtype``, ``StringDtype`` (incl. ``string[pyarrow]``),
    ``pd.ArrowDtype(pa.string())``, and similar extension dtypes — so callers
    can pass DataFrames produced by ``df.convert_dtypes()`` or a pyarrow CSV
    backend without first casting to ``object``/``category``.
    """
    return not pd.api.types.is_numeric_dtype(series)


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
        replaced wholesale, so partial overrides of BART hyperparameters
        must restate every key. Keys:

        - ``"bart"``: dict with ``m`` (int), ``alpha`` (float), ``beta`` (float).
        - ``"sigma"``: :class:`pymc_extras.prior.Prior` for the noise std.
        - ``"categorical_split"``: ``"onehot"`` (default) or ``"continuous"``.
          Controls how label-encoded categorical columns are split by BART
          — see Notes.
    sampler_config : dict, optional
        Passed to :func:`pymc.sample`. Defaults to ``{}``.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm import PIEModel, generate_synthetic_rct_corpus

        df = generate_synthetic_rct_corpus(n_campaigns=500, seed=42)
        X = df.drop(columns=["campaign_id", "measured_incrementality_per_dollar"])
        y = df["measured_incrementality_per_dollar"]

        model = PIEModel(
            pre_determined_features=[
                "objective",
                "vertical",
                "budget",
                "audience_type",
            ],
            post_determined_features=[
                "exposure_rate",
                "ctr",
                "last_click_conversions_per_dollar",
                "avg_treated_outcome",
            ],
        )
        model.fit(X, y, random_seed=42)
        preds = model.sample_posterior_predictive(X)

    Notes
    -----
    **This module is alpha — the API and defaults may change.** Tracked
    deviations from the paper (Gordon, Moakler & Zettelmeyer 2026):

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
    version = "0.1.0"

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
        self.pre_determined_features = list(pre_determined_features)
        self.post_determined_features = list(post_determined_features)
        self.target_column = target_column
        self._encoders: dict[str, LabelEncoder] = {}
        self._feature_columns: list[str] = []
        self._target_scale: float = 1.0

    @property
    def default_model_config(self) -> dict:
        """Default BART hyperparameters, noise prior, and categorical split mode.

        Returns
        -------
        dict
            Keys: ``"bart"`` (dict of scalars), ``"sigma"`` (Prior),
            ``"categorical_split"`` (str — ``"onehot"`` or ``"continuous"``).
        """
        return {
            "bart": {"m": 200, "alpha": 0.95, "beta": 2.0},
            "sigma": Prior("HalfNormal", sigma=1.0),
            "categorical_split": "onehot",
        }

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration (empty — PyMC auto-assigns PGBART + NUTS)."""
        return {}

    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        return {
            "bart": self.model_config["bart"],
            "sigma": self.model_config["sigma"],
            "categorical_split": self.model_config["categorical_split"],
        }

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
            Campaign feature matrix. Columns with ``object`` or ``category``
            dtype are label-encoded automatically.
        y : pd.Series or np.ndarray
            Measured incrementality (one value per campaign).
        """
        if pmb is None:
            raise ImportError(
                "pymc-bart is required for PIEModel. "
                "Install it with: pip install 'pymc-marketing[pie]'"
            )

        all_features = self.pre_determined_features + self.post_determined_features
        missing = set(all_features) - set(X.columns)
        if missing:
            raise ValueError(
                f"Features not found in X: {sorted(missing)}. "
                f"Available columns: {sorted(X.columns)}"
            )

        self._feature_columns = list(X.columns)
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

        cfg = self.model_config
        categorical_split = cfg.get("categorical_split", "onehot")
        if categorical_split not in ("onehot", "continuous"):
            raise ValueError(
                f"model_config['categorical_split'] must be 'onehot' or 'continuous', "
                f"got {categorical_split!r}."
            )
        if categorical_split == "onehot":
            split_rules = [
                OneHotSplitRule() if col in self._encoders else ContinuousSplitRule()
                for col in self._feature_columns
            ]
        else:
            split_rules = [ContinuousSplitRule() for _ in self._feature_columns]

        coords: dict[str, list] = {
            "obs": X.index.tolist(),
            "feature": X.columns.tolist(),
        }

        with pm.Model(coords=coords) as self.model:
            X_data = pm.Data(
                "X", X_encoded.values.astype(float), dims=("obs", "feature")
            )
            y_data = pm.Data("y_obs", y_scaled, dims="obs")
            # `Y` here is only used by BART for the leaf-prior `initval`
            # (frozen at fit time). The likelihood's `observed=y_data` is what
            # gets dummy-zeroed by `_data_setter` for out-of-sample prediction.
            mu = pmb.BART(
                "bart",
                X=X_data,
                Y=y_scaled,
                m=cfg["bart"]["m"],
                alpha=cfg["bart"]["alpha"],
                beta=cfg["bart"]["beta"],
                split_rules=split_rules,
                dims="obs",
            )
            sigma = cfg["sigma"].create_variable("sigma")
            pm.Normal(self.output_var, mu=mu, sigma=sigma, observed=y_data, dims="obs")

    def _data_setter(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        """Swap data inside the PyMC model for out-of-sample prediction.

        Parameters
        ----------
        X : pd.DataFrame
            New feature matrix. Must have the same columns as training X.
        y : pd.Series or np.ndarray, optional
            Target values. When ``None`` (predict mode), dummy zeros are used.
        """
        missing = set(self._feature_columns) - set(X.columns)
        extra = set(X.columns) - set(self._feature_columns)
        if missing or extra:
            raise ValueError(
                "X must contain exactly the training feature columns; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
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

        with self.model:
            pm.set_data(
                {
                    "X": X_encoded.values.astype(float),
                    "y_obs": y_data,
                },
                coords={"obs": X_encoded.index.tolist()},
            )

    def sample_posterior_predictive(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs: Any,
    ):
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

        with self.model:
            post_pred = pm.sample_posterior_predictive(self.idata, **kwargs)

        variable_name = (
            "predictions" if kwargs.get("predictions") else "posterior_predictive"
        )
        group = post_pred[variable_name]
        if self.output_var in group:
            group[self.output_var] = group[self.output_var] * self._target_scale

        if extend_idata:
            self.idata.extend(post_pred, join="right")  # type: ignore[union-attr]

        return az.extract(post_pred, variable_name, combined=combined)

    def build_from_idata(self, idata: az.InferenceData) -> None:
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
        if isinstance(kwargs.get("model_config", {}).get("sigma"), dict):
            kwargs["model_config"]["sigma"] = Prior.from_dict(
                kwargs["model_config"]["sigma"]
            )
        return kwargs
