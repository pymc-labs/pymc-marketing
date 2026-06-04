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
r"""Bass diffusion model for product adoption forecasting.

The recommended interface is :class:`BassModel` – a
:class:`~pymc_marketing.model_builder.ModelBuilder` subclass with
``.fit()``, ``.save()``, and ``.load()`` methods. The underlying
``pm.Model`` is accessible via ``model.model`` for users who want direct
access to the PyMC model object.

The standalone functions :func:`F`, :func:`f`, and :func:`create_bass_model`
are still exposed for direct use.

Adapted from Wiki: https://en.wikipedia.org/wiki/Bass_diffusion_model

The Bass diffusion model, developed by Frank Bass in 1969, is a mathematical model that describes
the process of how new products get adopted in a population over time. It is widely used in
marketing, forecasting, and innovation studies to predict the adoption rates of new products
and technologies.

Mathematical Formulation
------------------------
The model is based on a differential equation that describes the rate of adoption:

.. math::

    \frac{f(t)}{1-F(t)} = p + q F(t)

Where:

- :math:`F(t)` is the installed base fraction (cumulative proportion of adopters)
- :math:`f(t)` is the rate of change of the installed base fraction (:math:`f(t) = F'(t)`)
- :math:`p` is the coefficient of innovation or external influence
- :math:`q` is the coefficient of imitation or internal influence

The solution to this equation gives the adoption curve:

.. math::

    F(t) = \frac{1 - e^{-(p+q)t}}{1 + (\frac{q}{p})e^{-(p+q)t}}

The adoption rate at time t is given by:

.. math::

    f(t) = (p + q F(t))(1 - F(t))

Key Parameters
--------------
The model has three main parameters:

- :math:`m`: Market potential (total number of eventual adopters)
- :math:`p`: Coefficient of innovation (external influence) - typically 0.01-0.03
- :math:`q`: Coefficient of imitation (internal influence) - typically 0.3-0.5

Parameter Interpretation
------------------------
- A higher :math:`p` value indicates stronger external influence (advertising, marketing)
- A higher :math:`q` value indicates stronger internal influence (word-of-mouth, social interactions)
- The ratio :math:`q/p` indicates the relative strength of internal vs. external influences
- The peak of adoption occurs at time :math:`t^* = \frac{\ln(q/p)}{p+q}`

Applications
------------
The Bass model has been applied to forecast the adoption of various products and technologies:

- Consumer durables (TVs, refrigerators)
- Technology products (smartphones, software)
- Pharmaceutical products
- Entertainment products
- Services and subscriptions

This implementation provides a Bayesian version of the Bass model using PyMC, allowing for:
- Uncertainty quantification through prior distributions
- Hierarchical modeling for multiple products/markets
- Extension to incorporate additional factors

Examples
--------
Create a basic Bass model for multiple products:

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm

    from pymc_marketing.bass import create_bass_model, BassPriors
    from pymc_marketing.plot import plot_curve
    from pymc_extras.prior import Prior

    # Create time points - 3 years of monthly data
    n_dates = 12 * 3
    dates = pd.date_range(start="2020-01-01", periods=n_dates, freq="MS")
    t = np.arange(n_dates)

    # Define coordinates for multiple products
    coords = {"T": t, "product": ["A", "B", "C"]}

    # Define priors
    priors: BassPriors = {
        "m": Prior("DiracDelta", c=10_000),  # Market potential
        "p": Prior("Beta", alpha=13.85, beta=692.43, dims="product"),  # Innovation coefficient
        "q": Prior("Beta", alpha=36.2, beta=54.4),  # Imitation coefficient
        "likelihood": Prior("Poisson", dims=("T", "product")),
    }

    # Create the Bass model
    model = create_bass_model(t, observed=None, priors=priors, coords=coords)

    # Sample from the prior predictive distribution
    with model:
        idata = pm.sample_prior_predictive()

    # Plot the adoption curves
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    idata.prior["y"].pipe(plot_curve, "T", axes=axes)
    plt.suptitle("Bass Model Prior Predictive Adoption Curves")
    plt.tight_layout()
    plt.show()

"""

from typing import Any, TypedDict, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from numpy.typing import (
    ArrayLike,  # noqa: F401  # resolves pt.TensorLike's ForwardRef('ArrayLike') for sphinx_autodoc_typehints (#1197)
)
from pymc.model import Model
from pymc.util import RandomState
from pymc_extras.prior import Censored, Prior, VariableFactory, create_dim_handler

from pymc_marketing.bass.data import to_bass_dataset
from pymc_marketing.model_builder import ModelBuilder, create_sample_kwargs
from pymc_marketing.version import __version__


def F(
    p: float | pt.TensorVariable,
    q: float | pt.TensorVariable,
    t: float | pt.TensorVariable,
) -> pt.TensorVariable:
    r"""Installed base fraction (cumulative adoption proportion).

    This function calculates the cumulative proportion of adopters at time t,
    representing the fraction of the potential market that has adopted the product.

    Parameters
    ----------
    p : float or TensorVariable
        Coefficient of innovation (external influence)
    q : float or TensorVariable
        Coefficient of imitation (internal influence)
    t : array-like or TensorVariable
        Time points

    Returns
    -------
    TensorVariable
        The cumulative proportion of adopters at each time point

    Notes
    -----
    This is the solution to the Bass differential equation:

    .. math::

        F(t) = \frac{1 - e^{-(p+q)t}}{1 + (\frac{q}{p})e^{-(p+q)t}}

    When :math:`t=0`, :math:`F(t)=0`, and as :math:`t` approaches infinity, :math:`F(t)` approaches 1.
    """
    return (1 - pt.exp(-(p + q) * t)) / (1 + (q / p) * pt.exp(-(p + q) * t))


def f(
    p: float | pt.TensorVariable,
    q: float | pt.TensorVariable,
    t: float | pt.TensorVariable,
) -> pt.TensorVariable:
    r"""Installed base fraction rate of change (adoption rate).

    This function calculates the rate of new adoptions at time t as a
    proportion of the potential market. It represents the probability density
    function of adoption time.

    Parameters
    ----------
    p : float or TensorVariable
        Coefficient of innovation (external influence)
    q : float or TensorVariable
        Coefficient of imitation (internal influence)
    t : array-like or TensorVariable
        Time points

    Returns
    -------
    TensorVariable
        The adoption rate at each time point as a fraction of potential market

    Notes
    -----
    This is the derivative of F(t) with respect to time:

    .. math::

        f(t) = \frac{(p+q)^2 \cdot e^{-(p+q)t}}{p \cdot (1+\frac{q}{p}e^{-(p+q)t})^2}

    Alternatively:

    .. math::

        f(t) = (p + q \cdot F(t)) \cdot (1 - F(t))

    The peak adoption rate occurs at time :math:`t^* = \frac{\ln(q/p)}{p+q}`
    """
    return (p * pt.square(p + q) * pt.exp(t * (p + q))) / pt.square(
        p * pt.exp(t * (p + q)) + q
    )


class BassPriors(TypedDict):
    """Priors for the Bass diffusion model."""

    m: Prior | Censored | VariableFactory
    p: Prior | Censored | VariableFactory
    q: Prior | Censored | VariableFactory
    likelihood: Prior | Censored


def create_bass_model(
    t: pt.TensorLike,
    observed: pt.TensorLike | None,
    priors: BassPriors,
    coords: dict[str, Any],
    model: Model | None = None,
) -> Model:
    r"""Define a Bass diffusion model for product adoption forecasting.

    This function creates a Bayesian Bass diffusion model using PyMC to forecast
    product adoption over time. The Bass model captures both innovation (external
    influence like advertising) and imitation (internal influence like word-of-mouth)
    effects in the adoption process.

    The model includes the following components:

    - Market potential 'm': Total number of eventual adopters
    - Innovation coefficient 'p': Measures external influence
    - Imitation coefficient 'q': Measures internal influence
    - Adopters over time: Number of new adopters at each time point
    - Innovators: Adopters influenced by external factors
    - Imitators: Adopters influenced by previous adopters
    - Peak adoption time: When adoption rate reaches maximum

    Parameters
    ----------
    t : pt.TensorLike
        Time points for which the adoption is modeled.
    observed : pt.TensorLike | None
        Observed adoption data at each time point. If None, only
        prior predictive sampling is possible.
    priors : BassPriors
        Dictionary containing priors for:
        - 'm': Market potential prior
        - 'p': Innovation coefficient prior
        - 'q': Imitation coefficient prior
        - 'likelihood': Observation likelihood model
    coords : dict[str, Any]
        Coordinate values for dimensions in the model, including
        'date' for the time dimension and any other dimensions
        included in the prior specifications.
    model : Model, optional
        An existing PyMC model to use. If not provided, a new model is
        created with the given coords.

    Returns
    -------
    Model
        A PyMC model object for the Bass diffusion model, containing
        the variables m, p, q, adopters, innovators, imitators, peak,
        and the likelihood y.

    Notes
    -----
    The returned model can be used for prior predictive checks, posterior
    sampling, and posterior predictive checks to forecast product adoption.

    The model implements the following mathematical relationships:

    .. math::

        \text{adopters}(t) &= m \cdot f(p, q, t) \\
        \text{innovators}(t) &= m \cdot p \cdot (1 - F(p, q, t)) \\
        \text{imitators}(t) &= m \cdot q \cdot F(p, q, t) \cdot (1 - F(p, q, t)) \\
        \text{peak} &= \frac{\ln(q) - \ln(p)}{p + q}
    """
    model = model or pm.Model(coords=coords)
    with model:
        parameter_dims = (
            set(priors["p"].dims or ())
            .union(priors["q"].dims or ())
            .union(priors["m"].dims or ())
        )
        likelihood_dims = set(getattr(priors["likelihood"], "dims", ()) or ())

        combined_dims = (
            "T",
            *tuple(parameter_dims.union(likelihood_dims).difference(["T"])),
        )
        dim_handler = create_dim_handler(combined_dims)

        m = dim_handler(priors["m"].create_variable("m"), priors["m"].dims)
        p = dim_handler(priors["p"].create_variable("p"), priors["p"].dims)
        q = dim_handler(priors["q"].create_variable("q"), priors["q"].dims)

        time = dim_handler(t, "T")

        adopters = pm.Deterministic("adopters", m * f(p, q, time), dims=combined_dims)

        pm.Deterministic(
            "innovators",
            m * p * (1 - F(p, q, time)),
            dims=combined_dims,
        )
        pm.Deterministic(
            "imitators",
            m * q * F(p, q, time) * (1 - F(p, q, time)),
            dims=combined_dims,
        )

        peak = (pt.log(q) - pt.log(p)) / (p + q)
        peak_dims = tuple(parameter_dims) if parameter_dims else None
        pm.Deterministic("peak", peak, dims=peak_dims)

        priors["likelihood"].dims = combined_dims
        priors["likelihood"].create_likelihood_variable(  # type: ignore
            "y",
            mu=adopters,
            observed=observed,
        )

    return model


class BassModel(ModelBuilder):
    """Bass diffusion model for product adoption forecasting.

    Wraps the functional :func:`create_bass_model` inside the
    :class:`~pymc_marketing.model_builder.ModelBuilder` interface,
    providing standardised ``.fit()``, ``.save()``, ``.load()`` and
    related methods. The underlying ``pm.Model`` is accessible via
    ``model.model`` for direct use with PyMC functions.

    Parameters
    ----------
    model_config : dict, optional
        Dictionary with keys ``"m"``, ``"p"``, ``"q"``, ``"likelihood"``
        mapping to :class:`~pymc_extras.prior.Prior` (or equivalent dict).
        See :meth:`default_model_config` for defaults.
    sampler_config : dict, optional
        Dictionary of sampler settings (draws, tune, chains, …).
        See :meth:`default_sampler_config` for defaults.

    Notes
    -----
    Data format
    ~~~~~~~~~~~
    When using :class:`xr.Dataset`, the ``T`` coordinate is required and
    represents the time index. An ``observed`` data variable can hold
    adoption counts (omit for prior-predictive only).

    **Single-product** — 1-D ``observed`` with ``T`` as the only dimension:

    .. code-block:: python

        xr.Dataset(
            {"observed": ("T", counts)},
            coords={"T": np.arange(N)},
        )

    **Multi-product** — ``observed`` with ``T`` and ``product`` dimensions:

    .. code-block:: python

        xr.Dataset(
            {"observed": (("T", "product"), counts)},
            coords={"T": np.arange(N), "product": ["A", "B", "C"]},
        )

    Other input types (:class:`np.ndarray`, :class:`pd.Series`,
    :class:`pd.DataFrame`) are auto-converted via :func:`to_bass_dataset`.

    Examples
    --------
    **Fit a single-product model**

    .. code-block:: python

        import numpy as np
        import arviz as az
        from pymc_marketing.bass import BassModel

        y = np.random.poisson(lam=100, size=50)
        model = BassModel()
        idata = model.fit(data=y)
        print(az.summary(idata, var_names=["m", "p", "q"]))

    **Multi-product with custom priors**

    .. code-block:: python

        import xarray as xr
        from pymc_extras.prior import Prior

        data = xr.Dataset(
            {"observed": (("T", "product"), np.random.poisson(100, size=(50, 3)))},
            coords={"T": np.arange(50), "product": ["A", "B", "C"]},
        )
        model = BassModel(
            model_config={
                "m": Prior("Normal", mu=5_000, sigma=1_000),
                "p": Prior("Beta", alpha=1.5, beta=20),
                "q": Prior("Beta", alpha=2, beta=5),
                "likelihood": Prior("Poisson"),
            },
        )
        idata = model.fit(data=data)
        print(az.summary(idata, var_names=["m", "p", "q"]))

    **Generate synthetic data and fit**

    Build the model without an ``observed`` variable (only a ``T``
    coordinate), draw a prior predictive sample, then fit to it:

    .. code-block:: python

        import xarray as xr
        import pymc as pm

        ds = xr.Dataset({"T": np.arange(50)})
        model = BassModel()
        model.build_model(data=ds)

        with model.model:
            prior = pm.sample_prior_predictive(draws=50, random_seed=42)
            y_sim = prior.prior["y"].sel(draw=0, chain=0)

        # Now fit the model to the synthetic data
        idata = model.fit(data=y_sim.values)

    **Posterior predictive checks**

    Generate posterior predictive samples after fitting:

    .. code-block:: python

        pp_data = model.sample_posterior_predictive(X=new_time_points)

    The posterior contains deterministics such as ``adopters``,
    ``innovators``, ``imitators``, and ``peak`` that can be analysed
    directly via ``idata.posterior``, e.g.:

    .. code-block:: python

        az.plot_forest(idata.posterior["peak"], combined=True)
    """

    _model_type = "BassModel"
    version = __version__

    @property
    def default_model_config(self) -> dict:
        """Default model configuration with weakly informative priors."""
        return {
            "m": Prior("Normal", mu=0, sigma=10),
            "p": Prior("Beta", alpha=1.5, beta=20),
            "q": Prior("Beta", alpha=2, beta=5),
            "likelihood": Prior("Poisson"),
        }

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 4,
            "target_accept": 0.95,
        }

    @property
    def output_var(self) -> str:
        """Return the name of the output variable."""
        return "y"

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def _data_setter(
        self,
        X: xr.Dataset | pd.DataFrame | pd.Series | np.ndarray,
        y: np.ndarray | pd.Series | xr.DataArray | None = None,
    ) -> None:
        """Set new data in the model for posterior predictive sampling.

        Parameters
        ----------
        X : xr.Dataset, pd.DataFrame, pd.Series, np.ndarray
            New data, may have a different ``T`` length than the fitted
            model. If the data includes an ``observed`` variable, it will
            also be updated.
        y : optional
            Ignored; included for compatibility with ModelBuilder API.
        """
        ds = to_bass_dataset(X)
        new_t = ds.coords["T"].values
        set_data: dict[str, Any] = {"t": new_t}
        if "observed" in ds:
            set_data["y_obs"] = ds["observed"].values
        elif "y_obs" in self.model:
            dtype = self.model["y_obs"].get_value().dtype
            set_data["y_obs"] = np.zeros(len(new_t), dtype=dtype)
        with self.model:
            pm.set_data(set_data, coords={"T": new_t})

    def sample_posterior_predictive(
        self,
        X: xr.Dataset | pd.DataFrame | pd.Series | np.ndarray,
        extend_idata: bool = True,
        combined: bool = True,
        **sample_posterior_predictive_kwargs: Any,
    ) -> xr.Dataset:
        """Sample from the model's posterior predictive distribution.

        Parameters
        ----------
        X : xr.Dataset, pd.DataFrame, pd.Series, np.ndarray
            New data for prediction. Can have a different ``T`` length
            than the fitted data, enabling forecasting beyond the
            original time range.
        extend_idata : bool, optional
            Whether to add the predictions to ``self.idata``.
            Defaults to ``True``.
        combined : bool, optional
            Combine chain and draw dims into a single ``sample`` dim.
            Defaults to ``True``.
        **sample_posterior_predictive_kwargs
            Additional arguments passed to
            :func:`pymc.sample_posterior_predictive`.

        Returns
        -------
        xr.DataArray
            Posterior predictive samples.

        Examples
        --------
        **In-sample** (same number of time points, different t):

        .. code-block:: python

            pp = model.sample_posterior_predictive(X=new_t_data)

        **Out-of-sample forecast** (future time points):

        .. code-block:: python

            future = xr.Dataset({"T": np.arange(20, 30)})
            pp = model.sample_posterior_predictive(X=future)

        **Extended window** (past + future):

        .. code-block:: python

            extended = xr.Dataset({"T": np.arange(30)})
            pp = model.sample_posterior_predictive(X=extended)
        """
        self._data_setter(X)

        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )

        if extend_idata and self.idata is not None:
            self.idata.extend(post_pred, join="right")

        variable_name = (
            "predictions"
            if sample_posterior_predictive_kwargs.get("predictions")
            else "posterior_predictive"
        )

        return az.extract(post_pred, variable_name, combined=combined)

    def build_model(  # type: ignore[override]
        self,
        data: xr.Dataset | pd.DataFrame | pd.Series | np.ndarray | None = None,
    ) -> None:
        """Build the Bass diffusion model from the given data.

        After building, the underlying ``pm.Model`` is available via
        ``self.model``, giving direct access to all PyMC model
        functionality (e.g. ``pm.sample_prior_predictive``,
        ``pm.sample_posterior_predictive``).

        Parameters
        ----------
        data : optional
            Input data in one of the supported formats. If ``None``, reads
            from ``self.idata.fit_data`` (used internally by
            :meth:`build_from_idata`).
        """
        if data is not None:
            ds = to_bass_dataset(data)
        elif self.idata is not None and "fit_data" in self.idata:
            ds = self.idata.fit_data
        else:
            raise ValueError(
                "Data must be provided to build_model. "
                "Pass data directly or call build_model(data=...) first."
            )

        t = ds.coords["T"].values
        observed = ds.get("observed")

        coords = {name: ds.coords[name].values for name in ds.coords}

        self.model = pm.Model(coords=coords)

        with self.model:
            t_data = pm.Data("t", t, dims="T")
            if observed is not None:
                y_obs = pm.Data("y_obs", observed.values, dims=observed.dims)
            else:
                y_obs = None

            create_bass_model(
                t=t_data,
                observed=y_obs,
                priors=cast(BassPriors, self.model_config),
                coords=coords,
                model=self.model,
            )

    def fit(  # type: ignore[override]
        self,
        data: xr.Dataset | pd.DataFrame | pd.Series | np.ndarray,
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Fit the Bass diffusion model via MCMC.

        Parameters
        ----------
        data : xr.Dataset, pd.DataFrame, pd.Series, np.ndarray
            Adoption counts over time. See :func:`to_bass_dataset` for formats.
        progressbar : bool, optional
            Whether to show the progress bar. Defaults to ``True``.
        random_seed : optional
            Random seed for reproducibility.
        **kwargs
            Additional arguments forwarded to :func:`pymc.sample`.

        Returns
        -------
        arviz.InferenceData
            Posterior with parameters and deterministics (adopters,
            innovators, imitators, peak) plus a ``fit_data`` group.

        Notes
        -----
        After fitting, use standard ArviZ functions for posterior
        analysis:

        .. code-block:: python

            import arviz as az

            # Parameter summaries
            az.summary(idata, var_names=["m", "p", "q"])

            # Trace plots
            az.plot_trace(idata, var_names=["m", "p", "q"])

            # Forest plots of peak adoption time
            az.plot_forest(idata.posterior["peak"], combined=True)

        For posterior predictive sampling with new time points:

        .. code-block:: python

            pp = model.sample_posterior_predictive(X=new_data)
        """
        ds = to_bass_dataset(data)
        self.build_model(ds)

        sampler_kwargs = create_sample_kwargs(
            self.sampler_config, progressbar, random_seed, **kwargs
        )
        var_names = [v.name for v in self.model.free_RVs]

        with self.model:
            idata = pm.sample(var_names=var_names, **sampler_kwargs)
            idata.posterior = pm.compute_deterministics(
                idata.posterior, merge_dataset=True
            )

        if self.idata is not None:
            self.idata = self.idata.copy()
            self.idata.extend(idata, join="right")
        else:
            self.idata = idata

        self.idata["posterior"].attrs["pymc_marketing_version"] = __version__

        if "fit_data" in self.idata:
            del self.idata.fit_data

        self.idata.add_groups(fit_data=ds)
        self.set_idata_attrs(self.idata)
        return self.idata

    def build_from_idata(self, idata: az.InferenceData) -> None:
        """Rebuild the model from an ``InferenceData`` object.

        Used internally by :meth:`ModelBuilder.load`.
        """
        self.idata = idata
        self.build_model()
