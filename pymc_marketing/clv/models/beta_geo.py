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
"""Beta-Geometric Negative Binomial Distribution (BG/NBD) model for a non-contractual customer population across continuous time."""  # noqa: E501

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.xtensor as ptx
import xarray
from pymc.util import RandomState
from pymc_extras.prior import Prior, VariableFactory
from scipy.special import betaln, expit, hyp2f1

from pymc_marketing.clv.distributions import BetaGeoNBD
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray
from pymc_marketing.model_config import ModelConfig
from pymc_marketing.serialization import serialization
from pymc_marketing.terms import (
    Dot,
    ModelTerm,
    Named,
    Parameter,
    Product,
    Ref,
    Sum,
    Transform,
    _as_plain_tensor,
    _deserialize_child,
    build_param,
    collect_coords,
    collect_terms,
    register_data,
)


@serialization.register
@dataclass
class _Covariates(Named):
    """A covariate recipe carrying the DataFrame columns it consumes."""

    cols: Sequence[str] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the named recipe with its columns."""
        return {**super().to_dict(), "cols": list(self.cols)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _Covariates:
        """Reconstruct a covariate recipe from its serialized form."""
        return cls(
            name=data["name"],
            expr=_deserialize_child(data["expr"]),
            dims=data.get("dims"),
            cols=data.get("cols", ()),
        )


def _covariate_dataset(
    data: pd.DataFrame,
    *,
    purchase_cols: Sequence[str],
    dropout_cols: Sequence[str],
) -> xarray.Dataset:
    """Build the single dataset all covariate terms share."""
    data_vars = {}
    if purchase_cols:
        data_vars["purchase_data"] = (
            ("customer_id", "purchase_covariate"),
            data[purchase_cols].to_numpy(),
        )
    if dropout_cols:
        data_vars["dropout_data"] = (
            ("customer_id", "dropout_covariate"),
            data[dropout_cols].to_numpy(),
        )
    return xarray.Dataset(
        data_vars,
        coords={
            "customer_id": data["customer_id"].to_numpy(),
            "purchase_covariate": list(purchase_cols),
            "dropout_covariate": list(dropout_cols),
        },
    )


def create_purchase_covariates(
    purchase_covariate_cols: Sequence[str],
    *,
    scale_prior: VariableFactory | None = None,
    coefficient_prior: VariableFactory | None = None,
) -> _Covariates:
    """Create the standard BG/NBD purchase-rate covariate recipe.

    Composes ``alpha = alpha_scale * exp(-X @ beta)`` outside the model.
    The columns become part of the returned term, so passing this recipe
    replaces the deprecated ``purchase_covariate_cols`` config key.

    Parameters
    ----------
    purchase_covariate_cols : sequence of str
        DataFrame columns for the covariates.
    scale_prior : VariableFactory, optional
        Prior for ``alpha_scale``. Defaults to ``Prior("Weibull", alpha=2, beta=10)``.
    coefficient_prior : VariableFactory, optional
        Prior for the coefficients. Defaults to ``Prior("Normal", mu=0, sigma=1)``
        with ``dims="purchase_covariate"``.

    Returns
    -------
    _Covariates
        The ``alpha`` recipe with the columns embedded.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.clv.models.beta_geo import create_purchase_covariates

        alpha = create_purchase_covariates(["income", "age"])
        model = BetaGeoModel(model_config={"alpha": alpha})
    """
    cols = list(purchase_covariate_cols)
    scale = scale_prior or Prior("Weibull", alpha=2, beta=10)
    coefficient = coefficient_prior or Prior("Normal", mu=0, sigma=1)
    if hasattr(coefficient, "dims"):
        coefficient.dims = "purchase_covariate"
    expr = Parameter("alpha_scale", prior=scale, xdist=False) * Transform(
        -Dot(
            var_name="purchase_data",
            name="purchase_coefficient_alpha",
            prior=coefficient,
        ),
        func=ptx.math.exp,
    )
    return _Covariates("alpha", expr, dims="customer_id", cols=cols)


def create_dropout_covariates(
    dropout_covariate_cols: Sequence[str],
    *,
    a_prior: VariableFactory | None = None,
    b_prior: VariableFactory | None = None,
    coefficient_prior: VariableFactory | None = None,
) -> tuple[_Covariates, _Covariates]:
    """Create the standard nested BG/NBD dropout covariate recipes.

    Composes ``a = a_scale * exp(X @ beta_a)`` and ``b = b_scale * exp(X @ beta_b)``
    outside the model. The columns become part of the returned terms, so
    passing these recipes replaces the deprecated ``dropout_covariate_cols``
    config key.

    Parameters
    ----------
    dropout_covariate_cols : sequence of str
        DataFrame columns for the covariates.
    a_prior : VariableFactory, optional
        Prior for ``a_scale``. Defaults to ``Prior("Beta", alpha=2, beta=3)``.
    b_prior : VariableFactory, optional
        Prior for ``b_scale``. Defaults to ``Prior("Beta", alpha=3, beta=2)``.
    coefficient_prior : VariableFactory, optional
        Prior for the coefficients. Defaults to ``Prior("Normal", mu=0, sigma=1)``
        with ``dims="dropout_covariate"``.

    Returns
    -------
    tuple of _Covariates
        The ``a`` and ``b`` recipes, in that order.
    """
    cols = list(dropout_covariate_cols)
    a_scale = a_prior or Prior("Beta", alpha=2, beta=3)
    b_scale = b_prior or Prior("Beta", alpha=3, beta=2)
    coefficient = coefficient_prior or Prior("Normal", mu=0, sigma=1)
    if hasattr(coefficient, "dims"):
        coefficient.dims = "dropout_covariate"
    a_expr = Parameter("a_scale", prior=a_scale, xdist=False) * Transform(
        Dot(
            var_name="dropout_data",
            name="dropout_coefficient_a",
            prior=coefficient,
        ),
        func=ptx.math.exp,
    )
    b_expr = Parameter("b_scale", prior=b_scale, xdist=False) * Transform(
        Dot(
            var_name="dropout_data",
            name="dropout_coefficient_b",
            prior=coefficient,
        ),
        func=ptx.math.exp,
    )
    return (
        _Covariates("a", a_expr, dims="customer_id", cols=cols),
        _Covariates("b", b_expr, dims="customer_id", cols=cols),
    )


def _default_purchase_recipe(config: ModelConfig) -> ModelTerm:
    """Build the default ``alpha`` recipe from the model configuration."""
    cols = list(config.get("purchase_covariate_cols") or [])
    if cols:
        coefficient_prior: Any = config.get("purchase_coefficient") or Prior(
            "Normal", mu=0, sigma=1
        )
        coefficient_prior.dims = "purchase_covariate"
        expr = Parameter("alpha_scale", prior=config["alpha"]) * Transform(
            -Dot(
                var_name="purchase_data",
                name="purchase_coefficient_alpha",
                prior=coefficient_prior,
            ),
            func=ptx.math.exp,
        )
        return _Covariates("alpha", expr, dims="customer_id", cols=cols)
    return Parameter("alpha", prior=config["alpha"], xdist=False)


def _default_nested_dropout_recipes(config: ModelConfig) -> dict[str, ModelTerm]:
    """Build the default nested (``a``/``b``) dropout recipes from the model configuration."""
    cols = list(config.get("dropout_covariate_cols") or [])
    if cols:
        coefficient_prior: Any = config.get("dropout_coefficient") or Prior(
            "Normal", mu=0, sigma=1
        )
        coefficient_prior.dims = "dropout_covariate"
        a_expr = Parameter("a_scale", prior=config["a"]) * Transform(
            Dot(
                var_name="dropout_data",
                name="dropout_coefficient_a",
                prior=coefficient_prior,
            ),
            func=ptx.math.exp,
        )
        b_expr = Parameter("b_scale", prior=config["b"]) * Transform(
            Dot(
                var_name="dropout_data",
                name="dropout_coefficient_b",
                prior=coefficient_prior,
            ),
            func=ptx.math.exp,
        )
        return {
            "a": _Covariates("a", a_expr, dims="customer_id", cols=cols),
            "b": _Covariates("b", b_expr, dims="customer_id", cols=cols),
        }
    return {
        "a": Parameter("a", prior=config["a"], xdist=False),
        "b": Parameter("b", prior=config["b"], xdist=False),
    }


def _default_hierarchical_dropout_recipes(config: ModelConfig) -> dict[str, ModelTerm]:
    """Build the default hierarchical (``phi``/``kappa``) dropout recipes from the model configuration."""
    cols = list(config.get("dropout_covariate_cols") or [])
    recipes: dict[str, ModelTerm] = {
        "phi_dropout": Parameter(
            "phi_dropout", prior=config["phi_dropout"], xdist=False
        ),
        "kappa_dropout": Parameter(
            "kappa_dropout", prior=config["kappa_dropout"], xdist=False
        ),
    }
    if cols:
        coefficient_prior: Any = config.get("dropout_coefficient") or Prior(
            "Normal", mu=0, sigma=1
        )
        coefficient_prior.dims = "dropout_covariate"
        recipes["a_scale"] = Named("a_scale", Ref("phi_dropout") * Ref("kappa_dropout"))
        recipes["b_scale"] = Named(
            "b_scale", (1 - Ref("phi_dropout")) * Ref("kappa_dropout")
        )
        recipes["a"] = _Covariates(
            "a",
            Ref("a_scale")
            * Transform(
                Dot(
                    var_name="dropout_data",
                    name="dropout_coefficient_a",
                    prior=coefficient_prior,
                ),
                func=ptx.math.exp,
            ),
            dims="customer_id",
            cols=cols,
        )
        recipes["b"] = _Covariates(
            "b",
            Ref("b_scale")
            * Transform(
                Dot(
                    var_name="dropout_data",
                    name="dropout_coefficient_b",
                    prior=coefficient_prior,
                ),
                func=ptx.math.exp,
            ),
            dims="customer_id",
            cols=cols,
        )
    else:
        recipes["a"] = Named("a", Ref("phi_dropout") * Ref("kappa_dropout"))
        recipes["b"] = Named("b", (1 - Ref("phi_dropout")) * Ref("kappa_dropout"))
    return recipes


def _iter_dots(term: Any):
    """Yield every ``Dot`` reachable in a (possibly composed) term."""
    if isinstance(term, Dot):
        yield term
    elif isinstance(term, Sum):
        for child in term.terms:
            yield from _iter_dots(child)
    elif isinstance(term, Product):
        yield from _iter_dots(term.left)
        yield from _iter_dots(term.right)
    elif isinstance(term, Transform):
        yield from _iter_dots(term.inner)
    elif isinstance(term, Named):
        yield from _iter_dots(term.expr)


def _validate_modeling_dataset(
    ds: xarray.Dataset,
    *,
    purchase_cols: Sequence[str],
    dropout_cols: Sequence[str],
    data_bound: set[str],
) -> None:
    """Check a dataset has the variables the recipes need.

    Parameters
    ----------
    ds : xr.Dataset
        The modeling dataset.
    purchase_cols : sequence of str
        Effective purchase covariate columns.
    dropout_cols : sequence of str
        Effective dropout covariate columns.
    data_bound : set of str
        Data variable names bound by the recipes (``Dot.var_name`` values).

    Raises
    ------
    ValueError
        If required variables or coordinates are missing, if the dataset
        provides covariate data no recipe binds, or if the covariate
        coordinate labels do not match the effective columns.
    """
    required = {"customer_id", "T", "recency", "frequency"}
    if purchase_cols:
        required.add("purchase_data")
    if dropout_cols:
        required.add("dropout_data")
    missing = required - (set(ds.data_vars) | set(ds.coords))
    if missing:
        raise ValueError(
            "The dataset is missing required variables: "
            f"{sorted(missing)}. 'customer_id' must be a coordinate; "
            "'T', 'recency', and 'frequency' must be variables with the "
            "'customer_id' dimension."
        )
    if "customer_id" not in ds.coords:
        raise ValueError("The dataset must have 'customer_id' as a coordinate.")
    if (
        not purchase_cols
        and "purchase_data" in ds.data_vars
        and "purchase_data" not in data_bound
    ):
        raise ValueError(
            "The dataset provides 'purchase_data' but no purchase covariate "
            "columns are configured. Pass a purchase covariate recipe "
            "(create_purchase_covariates) or the deprecated "
            "'purchase_covariate_cols' config key."
        )
    if (
        not dropout_cols
        and "dropout_data" in ds.data_vars
        and "dropout_data" not in data_bound
    ):
        raise ValueError(
            "The dataset provides 'dropout_data' but no dropout covariate "
            "columns are configured. Pass a dropout covariate recipe "
            "(create_dropout_covariates) or the deprecated "
            "'dropout_covariate_cols' config key."
        )
    for dim, cols in (
        ("purchase_covariate", purchase_cols),
        ("dropout_covariate", dropout_cols),
    ):
        if cols and dim in ds.coords and list(ds.coords[dim].values) != list(cols):
            raise ValueError(
                f"The dataset's {dim!r} coordinate labels {list(ds.coords[dim].values)} "
                f"do not match the configured covariate columns {list(cols)}."
            )


def _dataset_to_dataframe(
    ds: xarray.Dataset,
    *,
    purchase_cols: Sequence[str],
    dropout_cols: Sequence[str],
) -> pd.DataFrame:
    """Reconstruct the modeling DataFrame from a modeling dataset.

    Prediction methods and serialization operate on DataFrames; the
    covariate columns are unwound from the ``purchase_data`` /
    ``dropout_data`` variables using their covariate coordinates.
    """
    data: dict[str, Any] = {
        "customer_id": np.asarray(ds.coords["customer_id"]),
        "T": np.asarray(ds["T"]),
        "recency": np.asarray(ds["recency"]),
        "frequency": np.asarray(ds["frequency"]),
    }
    for col, var, dim in [
        *[(col, "purchase_data", "purchase_covariate") for col in purchase_cols],
        *[(col, "dropout_data", "dropout_covariate") for col in dropout_cols],
    ]:
        data[col] = np.asarray(ds[var].sel({dim: col}))
    # Carry over remaining customer-level variables (e.g. extra user
    # columns such as dates) so the round-trip preserves them.
    for name in map(str, ds.data_vars):
        if name in data:
            continue
        values = np.asarray(ds[name])
        if values.ndim != 1:
            continue
        data[name] = values
    return pd.DataFrame(data)


class BetaGeoModel(CLVModel):
    r"""Beta-Geometric Negative Binomial Distribution (BG/NBD) model for a non-contractual customer population across continuous time.

    First introduced by Fader, Hardie & Lee [1]_, with additional predictive methods
    and enhancements in [2]_,[3]_, [4]_ and [5]_

    The BG/NBD model assumes dropout probabilities for the customer population are Beta distributed,
    and time between transactions follows a Gamma distribution while the customer is still active.

    This model requires data to be summarized by *recency*, *frequency*, and *T* for each customer,
    using `clv.utils.rfm_summary()` or equivalent. Modeling assumptions require *T >= recency*.

    Predictive methods have been adapted from the *BetaGeoFitter* class in the legacy ``lifetimes`` library
    (see https://github.com/CamDavidsonPilon/lifetimes/).

    Parameters
    ----------
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

        * ``customer_id``: Unique customer identifier
        * ``frequency``: Number of repeat purchases
        * ``recency``: Time between the first and the last purchase
        * ``T``: Time between the first purchase and the end of the observation period
    model_config : dict, optional
        Dictionary of model prior parameters:

        * ``alpha``: Scale parameter for time between purchases; defaults to ``Prior("Weibull", alpha=2, beta=10)``
        * ``r``: Shape parameter for time between purchases; defaults to ``Prior("Weibull", alpha=2, beta=1)``
        * ``a``: Shape parameter of dropout process; defaults to ``phi_purchase * kappa_purchase``
        * ``b``: Shape parameter of dropout process; defaults to ``(1 - phi_dropout) * kappa_dropout``
        * ``phi_dropout``: Nested prior for a and b priors; defaults to ``Prior("Uniform", lower=0, upper=1)``
        * ``kappa_dropout``: Nested prior for a and b priors; defaults to ``Prior("Pareto", alpha=1, m=1)``
        * ``purchase_covariates``: Coefficients for purchase rate covariates; defaults to ``Normal(0, 1)``
        * ``dropout_covariates``: Coefficients for dropout covariates; defaults to ``Normal.dist(0, 1)``
        * ``purchase_covariate_cols``: List containing column names of covariates for customer purchase rates.
        * ``dropout_covariate_cols``: List containing column names of covariates for customer dropouts.
    sampler_config : dict, optional
        Dictionary of sampler parameters. Defaults to *None*.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.clv import BetaGeoModel, rfm_summary

        # customer identifiers and purchase datetimes
        # are all that's needed to start modeling
        data = [
            [1, "2024-01-01"],
            [1, "2024-02-06"],
            [2, "2024-01-01"],
            [3, "2024-01-02"],
            [3, "2024-01-05"],
            [4, "2024-01-16"],
            [4, "2024-02-05"],
            [5, "2024-01-17"],
            [5, "2024-01-18"],
            [5, "2024-01-19"],
        ]
        raw_data = pd.DataFrame(data, columns=["id", "date"]

        # preprocess data
        rfm_df = rfm_summary(raw_data,'id','date')

        # model_config and sampler_configs are optional
        model = BetaGeoModel(
            model_config={
                "r": Prior("Weibull", alpha=2, beta=1),
                "alpha": Prior("HalfFlat"),
                "a": Prior("Beta", alpha=2, beta=3),
                "b": Prior("Beta", alpha=3, beta=2),
            },
            sampler_config={
                "draws": 1000,
                "tune": 1000,
                "chains": 2,
                "cores": 2,
            },
        )

        # The default 'mcmc' fit_method provides informative predictions
        # and reliable performance on small datasets
        model.fit(data=rfm_df)
        print(model.fit_summary())

        # Maximum a Posteriori can quickly fit a model to large datasets,
        # but will give limited insights into predictive uncertainty.
        model.fit(fit_method='map')
        print(model.fit_summary())

        # Predict number of purchases for current customers
        # over the next 10 time periods
        expected_purchases = model.expected_purchases(future_t=10)

        # Predict probability customers are still active
        probability_alive = model.expected_probability_alive()

        # Predict number of purchases for a new customer over 't' time periods
        expected_purchases_new_customer = model.expected_purchases_new_customer(t=10)

    References
    ----------
    .. [1] Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). “Counting your customers
           the easy way: An alternative to the Pareto/NBD model." Marketing science,
           24(2), 275-284. http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
    .. [2] Fader, P. S., Hardie, B. G., & Lee, K. L. (2008). "Computing
           P (alive) using the BG/NBD model." http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf.
    .. [3] Fader, P. S. & Hardie, B. G. (2013) "Overcoming the BG/NBD Model's #NUM!
           Error Problem." http://brucehardie.com/notes/027/bgnbd_num_error.pdf.
    .. [4] Fader, P. S. & Hardie, B. G. (2019) "A Step-by-Step Derivation of the BG/NBD
           Model." https://www.brucehardie.com/notes/039/bgnbd_derivation__2019-11-06.pdf
    .. [5] Fader, Peter & G. S. Hardie, Bruce (2007).
           "Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD Models".
           https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf

    """  # noqa: E501

    _model_type = "BG/NBD"  # Beta-Geometric Negative Binomial Distribution
    _skipped_config_keys = {
        "a",
        "b",
        "purchase_covariate_cols",
        "dropout_covariate_cols",
        "purchase_coefficient",
        "dropout_coefficient",
    }

    def __init__(
        self,
        *,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        super().__init__(
            model_config=model_config,
            sampler_config=sampler_config,
        )
        self._warn_deprecated_covariate_config()
        self._resolve_parameter_recipes()

    _DEPRECATED_CONFIG_KEYS = {
        "purchase_covariate_cols": (
            "Pass a term recipe instead, e.g. model_config={'alpha': "
            "create_purchase_covariates(['cov'])}. The columns become part "
            "of the term and are serialized with it."
        ),
        "dropout_covariate_cols": (
            "Pass term recipes instead, e.g. model_config={'a': recipe, "
            "'b': recipe} from create_dropout_covariates(['cov']). The "
            "columns become part of the terms and are serialized with them."
        ),
        "purchase_coefficient": (
            "Pass coefficient_prior= to create_purchase_covariates instead; "
            "the prior becomes part of the serialized recipe."
        ),
        "dropout_coefficient": (
            "Pass coefficient_prior= to create_dropout_covariates instead; "
            "the prior becomes part of the serialized recipe."
        ),
    }

    def _warn_deprecated_covariate_config(self) -> None:
        """Warn when deprecated covariate configuration keys are used."""
        for key, guidance in self._DEPRECATED_CONFIG_KEYS.items():
            if self.model_config.get(key) or key in self.model_config:
                warnings.warn(
                    f"{key!r} in model_config is deprecated and will be removed "
                    f"in a future release. {guidance}",
                    DeprecationWarning,
                    stacklevel=3,
                )

    def _resolve_parameter_recipes(self) -> None:
        """Resolve the parameter terms for this instance, outside any PyMC context."""
        config = self.model_config

        alpha = config["alpha"]
        self._alpha_recipe: ModelTerm = (
            alpha if isinstance(alpha, ModelTerm) else _default_purchase_recipe(config)
        )

        a, b = config.get("a"), config.get("b")
        if isinstance(a, ModelTerm) != isinstance(b, ModelTerm):
            raise ValueError("Provide both 'a' and 'b' term recipes or neither.")
        if isinstance(a, ModelTerm) and isinstance(b, ModelTerm):
            self._dropout_recipes = {"a": a, "b": b}
        elif "a" in config and "b" in config:
            self._dropout_recipes = _default_nested_dropout_recipes(config)
        else:
            self._dropout_recipes = _default_hierarchical_dropout_recipes(config)

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration.

        The coefficient priors are part of the parameter recipes (and their
        serialization); the deprecated ``purchase_coefficient`` /
        ``dropout_coefficient`` keys still feed the legacy covariate path.
        """
        return {
            "alpha": Prior("Weibull", alpha=2, beta=10),
            "r": Prior("Weibull", alpha=2, beta=1),
            "phi_dropout": Prior("Uniform", lower=0, upper=1),
            "kappa_dropout": Prior("Pareto", alpha=1, m=1),
        }

    @property
    def _parameter_spec(self) -> dict[str, ModelTerm]:
        """Ordered parameter terms for ``build_model``."""
        spec: dict[str, ModelTerm] = {
            "alpha": self._alpha_recipe,
            **self._dropout_recipes,
        }
        spec["r"] = Parameter("r", prior=self.model_config["r"], xdist=False)
        return spec

    @property
    def purchase_covariate_cols(self) -> list[str]:
        """Purchase covariate column names from the recipe or model_config."""
        if isinstance(self._alpha_recipe, _Covariates):
            return list(self._alpha_recipe.cols)
        return list(self.model_config.get("purchase_covariate_cols", []))

    @property
    def dropout_covariate_cols(self) -> list[str]:
        """Dropout covariate column names from the recipes or model_config."""
        for recipe in self._dropout_recipes.values():
            if isinstance(recipe, _Covariates) and recipe.cols:
                return list(recipe.cols)
        return list(self.model_config.get("dropout_covariate_cols", []))

    @property
    def covariate_cols(self) -> list[str]:
        """All covariate column names."""
        return self.purchase_covariate_cols + self.dropout_covariate_cols

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate BG/NBD-specific data requirements."""
        super()._validate_data(data)
        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                "frequency",
                "recency",
                "T",
                *self.covariate_cols,
            ],
            must_be_unique=["customer_id"],
        )

    def build_model(self, data: pd.DataFrame | xarray.Dataset) -> None:  # type: ignore[override]
        """Build the model.

        The parameter terms are composed outside the model context (see
        ``_resolve_parameter_recipes``); this method only registers their
        data, builds the variables, and attaches the likelihood.

        Parameters
        ----------
        data : pd.DataFrame or xr.Dataset
            Input data with customer_id, frequency, recency, and T columns.
            A dataset may be passed instead. It must provide ``T``,
            ``recency``, and ``frequency`` variables with the
            ``customer_id`` coordinate, plus ``purchase_data`` /
            ``dropout_data`` variables when the recipes use covariates
            (covariate coordinates must match the configured columns). A
            DataFrame is reconstructed from the dataset, so prediction
            methods keep working on columns.

        Raises
        ------
        ValueError
            If the dataset is missing required variables, provides
            covariate data without configured columns, or its covariate
            coordinates do not match the configured columns.
        """
        spec = self._parameter_spec
        terms = collect_terms(list(spec.values()))
        data_bound = {
            dot.var_name for term in spec.values() for dot in _iter_dots(term)
        }

        spec = self._parameter_spec
        terms = collect_terms(list(spec.values()))
        data_bound = {
            dot.var_name for term in spec.values() for dot in _iter_dots(term)
        }

        if isinstance(data, xarray.Dataset):
            ds = data
            _validate_modeling_dataset(
                ds,
                purchase_cols=self.purchase_covariate_cols,
                dropout_cols=self.dropout_covariate_cols,
                data_bound=data_bound,
            )
            self.data = _dataset_to_dataframe(
                ds,
                purchase_cols=self.purchase_covariate_cols,
                dropout_cols=self.dropout_covariate_cols,
            )
            self._validate_data(self.data)
        else:
            self._validate_data(data)
            self.data = data
            self._check_dataframe_recipe_support(data_bound)
            ds = _covariate_dataset(
                self.data,
                purchase_cols=self.purchase_covariate_cols,
                dropout_cols=self.dropout_covariate_cols,
            )
        # Persist the dataset the recipes actually bound so it survives
        # serialization (see ``create_fit_data_group``).
        self._modeling_data = ds

        coords = {
            "purchase_covariate": self.purchase_covariate_cols,
            "dropout_covariate": self.dropout_covariate_cols,
            "customer_id": self.data["customer_id"],
            "obs_var": ["recency", "frequency"],
        }
        coords = {**coords, **collect_coords(*terms, ds=ds)}

        with pm.Model(coords=coords) as self.model:
            for term in terms:
                register_data(term, ds=ds)

            built = {name: build_param(term) for name, term in spec.items()}

            BetaGeoNBD(
                name="recency_frequency",
                alpha=_as_plain_tensor(built["alpha"]),
                a=_as_plain_tensor(built["a"]),
                b=_as_plain_tensor(built["b"]),
                r=_as_plain_tensor(built["r"]),
                T=self.data["T"],
                observed=np.stack(
                    (self.data["recency"], self.data["frequency"]), axis=1
                ),
                dims=["customer_id", "obs_var"],
            )

    def _check_dataframe_recipe_support(self, data_bound: set[str]) -> None:
        """Raise for recipes the DataFrame path cannot data-bind.

        The DataFrame path can construct exactly two data variables: from
        the purchase covariate columns (``purchase_data``) and the dropout
        covariate columns (``dropout_data``). Any other recipe-bound
        variable requires an ``xr.Dataset`` passed to ``build_model``.
        """
        can_provide = {
            "purchase_data": bool(self.purchase_covariate_cols),
            "dropout_data": bool(self.dropout_covariate_cols),
        }
        for var_name in sorted(data_bound):
            if not can_provide.get(var_name, False):
                raise ValueError(
                    f"A recipe binds the data variable {var_name!r} which cannot "
                    "be constructed from a DataFrame. Pass the covariate columns "
                    "(create_purchase_covariates / create_dropout_covariates), or "
                    "an xr.Dataset to build_model."
                )

    def _check_recipes_predictable(self) -> None:
        """Raise for parameter recipes without a predictive evaluation."""
        for name, recipe in [
            ("alpha", self._alpha_recipe),
            *self._dropout_recipes.items(),
        ]:
            if isinstance(recipe, (Parameter, _Covariates)):
                continue
            if not any(True for _ in _iter_dots(recipe)):
                continue
            raise NotImplementedError(
                f"Predictive methods are not implemented for the {name!r} recipe "
                f"({type(recipe).__name__}). Use the default recipes, "
                "create_purchase_covariates, or create_dropout_covariates."
            )

    def _extract_predictive_variables(
        self,
        data: pd.DataFrame,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        """
        Extract predictive variables from the data.

        Utility function assigning default customer arguments for predictive methods and converting to xarrays.
        """
        self._check_recipes_predictable()
        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                *customer_varnames,
                *self.purchase_covariate_cols,
                *self.dropout_covariate_cols,
            ],
            must_be_unique=["customer_id"],
        )

        customer_id = data["customer_id"]
        model_coords = self.model.coords
        if self.purchase_covariate_cols:
            purchase_xarray = xarray.DataArray(
                data[self.purchase_covariate_cols],
                dims=["customer_id", "purchase_covariate"],
                coords=[customer_id, list(model_coords["purchase_covariate"])],
            )
            alpha_scale = self.fit_result["alpha_scale"]
            purchase_coefficient_alpha = self.fit_result["purchase_coefficient_alpha"]
            alpha = alpha_scale * np.exp(
                -xarray.dot(
                    purchase_xarray,
                    purchase_coefficient_alpha,
                    dim="purchase_covariate",
                )
            )
            alpha.name = "alpha"
        else:
            alpha = self.fit_result["alpha"]

        if self.dropout_covariate_cols:
            dropout_xarray = xarray.DataArray(
                data[self.dropout_covariate_cols],
                dims=["customer_id", "dropout_covariate"],
                coords=[customer_id, list(model_coords["dropout_covariate"])],
            )
            a_scale = self.fit_result["a_scale"]
            dropout_coefficient_a = self.fit_result["dropout_coefficient_a"]
            a = a_scale * np.exp(
                xarray.dot(
                    dropout_xarray, dropout_coefficient_a, dim="dropout_covariate"
                )
            )
            a.name = "a"

            dropout_coefficient_b = self.fit_result["dropout_coefficient_b"]
            b_scale = self.fit_result["b_scale"]
            b = b_scale * np.exp(
                xarray.dot(
                    dropout_xarray, dropout_coefficient_b, dim="dropout_covariate"
                )
            )
            b.name = "b"
        else:
            a = self.fit_result["a"]
            b = self.fit_result["b"]

        r = self.fit_result["r"]

        customer_vars = to_xarray(
            data["customer_id"],
            *[data[customer_varname] for customer_varname in customer_varnames],
        )
        if len(customer_varnames) == 1:
            customer_vars = [customer_vars]

        return xarray.combine_by_coords(
            (
                a,
                b,
                alpha,
                r,
                *customer_vars,
            ),
            compat="override",
        )

    def expected_purchases(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute the expected number of future purchases across *future_t* time periods given *recency*, *frequency*, and *T* for each customer.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from equation (10) in [1]_, and the legacy ``lifetimes`` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L201

        Parameters
        ----------
        future_t : int, array_like
            Number of time periods to predict expected purchases.
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Time between the first and the last purchase
            * `T`: Time between first purchase and end of observation period; model assumptions require T >= recency

        References
        ----------
        .. [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
            "Counting Your Customers the Easy Way: An Alternative to the
            Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
            https://www.brucehardie.com/papers/bgnbd_2004-04-20.pdf

        """  # noqa: E501
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T", "future_t"]
        )
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]
        t = dataset["future_t"]

        numerator = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(
            r + x,
            b + x,
            a + b + x - 1,
            t / (alpha + T + t),
        )
        numerator *= (a + b + x - 1) / (a - 1)
        denominator = 1 + (x > 0) * (a / (b + x - 1)) * (
            (alpha + T) / (alpha + t_x)
        ) ** (r + x)

        return (numerator / denominator).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        r"""Compute the probability a customer with history *frequency*, *recency*, and *T* is currently active.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from page (2) in Bruce Hardie's notes [1]_, and the legacy ``lifetimes`` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L260

        Parameters
        ----------
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:

            * ``customer_id``: Unique customer identifier
            * ``frequency``: Number of repeat purchases
            * ``recency``: Time between the first and the last purchase
            * ``T``: Time between first purchase and end of observation period, model assumptions require T >= recency

        References
        ----------
        .. [1] Fader, P. S., Hardie, B. G., & Lee, K. L. (2008). Computing
               P (alive) using the BG/NBD model. http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf.

        """
        if data is None:
            data = self.data

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T"]
        )
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]

        log_div = (r + x) * np.log((alpha + T) / (alpha + t_x)) + np.log(
            a / (b + np.maximum(x, 1) - 1)
        )

        return xarray.where(x == 0, 1.0, expit(-log_div)).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_no_purchase(
        self,
        t: int,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        r"""Compute the probability a customer with history frequency, recency, and T
        will have 0 purchases in the period (T, T+t].

        The data parameter is only required for out-of-sample customers.

        Adapted from Section 5.3, Equation 34 in Bruce Hardie's notes [1]_.

        Parameters
        ----------
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:

            * ``customer_id``: Unique customer identifier
            * ``frequency``: Number of repeat purchases
            * ``recency``: Time between the first and the last purchase
            * ``T``: Time between first purchase and end of observation period, model assumptions require T >= recency

        t : int
            Days after T which defines the range (T, T+t].

        References
        ----------
        .. [1] Fader, P. S. & Hardie, B. G. (2019) "A Step-by-Step Derivation of the
                BG/NBD Model." https://www.brucehardie.com/notes/039/bgnbd_derivation__2019-11-06.pdf
        """  # noqa: D205
        if data is None:
            data = self.data

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T"]
        )
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]

        E = alpha + t_x
        F = alpha + T + t
        M = alpha + T

        beta_rep = betaln(a, b + x)
        K_E = betaln(a + 1, b + x - 1) - (r + x) * np.log(E)
        K_F = beta_rep - (r + x) * np.log(F)
        K_M = beta_rep - (r + x) * np.log(M)

        K1 = np.maximum(K_E, K_F)
        K2 = np.maximum(K_E, K_M)

        numer = np.exp(K_E - K1) + np.exp(K_F - K1)
        denom = np.exp(K_E - K2) + np.exp(K_M - K2)

        prob_no_deposits = np.exp(K1 - K2) * numer / denom

        return prob_no_deposits.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute the expected number of purchases for a new customer across *t* time periods.

        Adapted from equation (9) in [1]_, and the legacy ``lifetimes`` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L328

        Parameters
        ----------
        t : array_like
            Number of time periods over which to estimate purchases.

        References
        ----------
        .. [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
            "Counting Your Customers the Easy Way: An Alternative to the
            Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
            http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf

        """
        # TODO: This is extraneous now, but needed for future covariate support.
        if data is None:
            data = self.data

        if t is not None:
            data = data.assign(t=t)

        dataset = self._extract_predictive_variables(data, customer_varnames=["t"])
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        t = dataset["t"]

        first_term = (a + b - 1) / (a - 1)
        second_term = 1 - (alpha / (alpha + t)) ** r * hyp2f1(
            r, b, a + b - 1, t / (alpha + t)
        )

        return (first_term * second_term).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def distribution_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        T: int | np.ndarray | pd.Series | None = None,
        random_seed: RandomState | None = None,
        var_names: Sequence[
            Literal["dropout", "purchase_rate", "recency_frequency"]
        ] = ("dropout", "purchase_rate", "recency_frequency"),
        n_samples: int = 1000,
    ) -> xarray.Dataset:
        """Compute posterior predictive samples of dropout, purchase rate and frequency/recency of new customers.

        In a model with covariates, if `data` is not specified, the dataset used for fitting will be used and
        a prediction will be computed for a *new customer* with each set of covariates.
        *This is not a conditional prediction for observed customers!*

        Parameters
        ----------
        data : ~pandas.DataFrame, Optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `T`: Time between the first purchase and the end of the observation period

            If not provided, predictions will be ran with data used to fit model.
        T : array_like, optional
            time between the first purchase and the end of the observation period.
            Not needed if `data` parameter is provided with a `T` column.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.
        var_names : sequence of str, optional
            Names of the variables to sample from. Defaults to ["dropout", "purchase_rate", "recency_frequency"].
        n_samples : int, optional
            Number of samples to generate. Defaults to 1000

        """
        if data is None:
            data = self.data

        if T is not None:
            data = data.assign(T=T)

        dataset = self._extract_predictive_variables(data, customer_varnames=["T"])
        T = dataset["T"].values
        # Delete "T" so we can pass dataset directly to `sample_posterior_predictive`
        del dataset["T"]

        if dataset.sizes["chain"] == 1 and dataset.sizes["draw"] == 1:
            # For map fit add a dummy draw dimension
            dataset = dataset.squeeze("draw").expand_dims(draw=range(n_samples))

        coords = self.model.coords.copy()  # type: ignore
        coords["customer_id"] = data["customer_id"]

        with pm.Model(coords=coords) as pred_model:
            if self.purchase_covariate_cols:
                alpha = pm.Flat("alpha", dims=["customer_id"])
            else:
                alpha = pm.Flat("alpha")

            if self.dropout_covariate_cols:
                a = pm.Flat("a", dims=["customer_id"])
                b = pm.Flat("b", dims=["customer_id"])
            else:
                a = pm.Flat("a")
                b = pm.Flat("b")

            r = pm.Flat("r")

            pm.Beta(
                "dropout", alpha=a, beta=b, dims=pred_model.named_vars_to_dims.get("a")
            )
            pm.Gamma(
                "purchase_rate",
                alpha=r,
                beta=alpha,
                dims=pred_model.named_vars_to_dims.get("alpha"),
            )

            BetaGeoNBD(
                name="recency_frequency",
                a=a,
                b=b,
                r=r,
                alpha=alpha,
                T=T,
                dims=["customer_id", "obs_var"],
            )

            return pm.sample_posterior_predictive(
                dataset,
                var_names=var_names,
                random_seed=random_seed,
                predictions=True,
            ).predictions

    def distribution_new_customer_dropout(
        self,
        data: pd.DataFrame | None = None,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        """Sample the Beta distribution for the population-level dropout rate.

        This is the probability that a new customer will "drop out" and make no further purchases.

        Parameters
        ----------
        random_seed : RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        xarray.Dataset
            Dataset containing the posterior samples for the population-level dropout rate.

        """
        return self.distribution_new_customer(
            data=data,
            random_seed=random_seed,
            var_names=["dropout"],
        )["dropout"]

    def distribution_new_customer_purchase_rate(
        self,
        data: pd.DataFrame | None = None,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        """Sample the Gamma distribution for the population-level purchase rate.

        This is the purchase rate for a new customer and determines the time between
        purchases for any new customer.

        Parameters
        ----------
        random_seed : RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        xarray.Dataset
            Dataset containing the posterior samples for the population-level purchase rate.

        """
        return self.distribution_new_customer(
            data=data,
            random_seed=random_seed,
            var_names=["purchase_rate"],
        )["purchase_rate"]

    def distribution_new_customer_recency_frequency(
        self,
        data: pd.DataFrame | None = None,
        *,
        T: int | np.ndarray | pd.Series | None = None,
        random_seed: RandomState | None = None,
        n_samples: int = 1000,
    ) -> xarray.Dataset:
        """BG/NBD process representing purchases across the customer population.

        This is the distribution of purchase frequencies given 'T' observation periods for each customer.

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `T`: Time between the first purchase and the end of the observation period.
            * All covariate columns specified when model was initialized.

            If not provided, the method will use the fit dataset.
        T : array_like, optional
            Number of observation periods for each customer. If not provided, T values from fit dataset will be used.
            Not required if `data` Dataframe contains a `T` column.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.
        n_samples : int, optional
            Number of samples to generate. Defaults to 1000.

        Returns
        -------
        ~xarray.Dataset
            Dataset containing the posterior samples for the customer population.

        """
        return self.distribution_new_customer(
            data=data,
            T=T,
            random_seed=random_seed,
            var_names=["recency_frequency"],
            n_samples=n_samples,
        )["recency_frequency"]
