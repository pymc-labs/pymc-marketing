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

from inspect import signature
from typing import Any, TypedDict, cast

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import (
    ArrayLike,  # noqa: F401  # resolves pt.TensorLike's ForwardRef('ArrayLike') for sphinx_autodoc_typehints (#1197)
)
from pymc.model import Model
from pymc.util import RandomState
from pymc_extras.prior import (
    Censored,
    MuAlreadyExistsError,
    Prior,
    UnsupportedDistributionError,
    VariableFactory,
)
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.bass import plotting
from pymc_marketing.bass.data import to_bass_dataset
from pymc_marketing.model_builder import ModelBuilder, SamplingMethod
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.version import __version__


def _exp(
    x: float | pt.TensorVariable | XTensorVariable,
) -> pt.TensorVariable | XTensorVariable:
    """``exp`` that works for floats, PyTensor tensors, and xtensor variables.

    Lets :func:`F` and :func:`f` be used both with NumPy/float inputs and with
    the named-dims (xtensor) variables of the ``pymc.dims`` model graph.
    """
    return pmd.math.exp(x) if isinstance(x, XTensorVariable) else pt.exp(x)


def F(
    p: float | pt.TensorVariable | XTensorVariable,
    q: float | pt.TensorVariable | XTensorVariable,
    t: float | pt.TensorVariable | XTensorVariable,
) -> pt.TensorVariable | XTensorVariable:
    r"""Installed base fraction (cumulative adoption proportion).

    This function calculates the cumulative proportion of adopters at time t,
    representing the fraction of the potential market that has adopted the product.

    Parameters
    ----------
    p : float, TensorVariable or XTensorVariable
        Coefficient of innovation (external influence)
    q : float, TensorVariable or XTensorVariable
        Coefficient of imitation (internal influence)
    t : array-like, TensorVariable or XTensorVariable
        Time points

    Returns
    -------
    TensorVariable or XTensorVariable
        The cumulative proportion of adopters at each time point

    Notes
    -----
    This is the solution to the Bass differential equation:

    .. math::

        F(t) = \frac{1 - e^{-(p+q)t}}{1 + (\frac{q}{p})e^{-(p+q)t}}

    When :math:`t=0`, :math:`F(t)=0`, and as :math:`t` approaches infinity, :math:`F(t)` approaches 1.
    """
    return (1 - _exp(-(p + q) * t)) / (1 + (q / p) * _exp(-(p + q) * t))


def f(
    p: float | pt.TensorVariable | XTensorVariable,
    q: float | pt.TensorVariable | XTensorVariable,
    t: float | pt.TensorVariable | XTensorVariable,
) -> pt.TensorVariable | XTensorVariable:
    r"""Installed base fraction rate of change (adoption rate).

    This function calculates the rate of new adoptions at time t as a
    proportion of the potential market. It represents the probability density
    function of adoption time.

    Parameters
    ----------
    p : float, TensorVariable or XTensorVariable
        Coefficient of innovation (external influence)
    q : float, TensorVariable or XTensorVariable
        Coefficient of imitation (internal influence)
    t : array-like, TensorVariable or XTensorVariable
        Time points

    Returns
    -------
    TensorVariable or XTensorVariable
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
    exp_t = _exp(t * (p + q))
    return (p * (p + q) ** 2 * exp_t) / (p * exp_t + q) ** 2


def _supports_xdist(prior: Prior | Censored | VariableFactory) -> bool:
    """Whether ``prior`` can build itself directly in ``pymc.dims`` space.

    The check reads the ``create_variable`` signature for a literal ``xdist``
    parameter, so a factory that hides it behind ``**kwargs`` is treated as
    not supporting it and gets the ``as_xtensor`` wrapping instead. Declare
    the parameter explicitly to take the native path.
    """
    if "xdist" not in signature(prior.create_variable).parameters:
        # A VariableFactory written against the ``create_variable(name)``
        # signature the protocol documents.
        return False

    if isinstance(prior, Censored):
        return _supports_xdist(prior.distribution)

    if isinstance(prior, Prior):
        return hasattr(pmd, prior.distribution)

    # A factory that takes the kwarg handles its own dispatch.
    return True


def _create_dim_variable(
    prior: Prior | Censored | VariableFactory, name: str
) -> XTensorVariable:
    """Create a named-dims (xtensor) variable for a prior.

    Uses ``xdist=True`` so the variable lives in xtensor space from the start.
    Whatever cannot take that route -- a distribution ``pymc.dims`` does not
    implement (``Wald``, ...) or a custom
    factory written against the ``create_variable(name)`` signature -- gets a
    regular variable wrapped with ``pmd.as_xtensor``.
    """
    if not _supports_xdist(prior):
        return pmd.as_xtensor(prior.create_variable(name), dims=prior.dims or ())
    return prior.create_variable(name, xdist=True)


def _create_likelihood_variable(
    prior: Prior | Censored,
    name: str,
    mu: XTensorVariable,
    observed: XTensorVariable | None,
    dims: tuple[str, ...],
) -> pt.TensorVariable | XTensorVariable:
    """Create the outcome variable, observed or not.

    A distribution ``pymc.dims`` does not implement gets the regular pymc
    path, on the underlying tensors in ``dims`` order.

    ``create_likelihood_variable`` is for the observed case only: a
    likelihood needs data, so pymc_extras refuses ``observed=None`` there
    (pymc-devs/pymc-extras#731). Prior predictive still needs the outcome
    node, so build it with ``create_variable`` and ``mu`` attached, keeping
    the same guards the pymc_extras method applies.
    """
    xdist = _supports_xdist(prior)

    if observed is not None:
        if xdist:
            return prior.create_likelihood_variable(
                name, mu=mu, observed=observed, xdist=True
            )
        return prior.create_likelihood_variable(
            name,
            mu=mu.transpose(*dims).values,
            observed=observed.transpose(*dims).values,
        )

    # Censored keeps its parameters on the wrapped distribution.
    inner = prior.distribution if isinstance(prior, Censored) else prior
    if "mu" not in signature(inner.pymc_distribution.dist).parameters:
        raise UnsupportedDistributionError(
            f"Likelihood distribution {inner.distribution!r} is not supported."
        )
    if "mu" in inner.parameters:
        raise MuAlreadyExistsError(inner)

    unobserved = inner.deepcopy()
    unobserved.parameters["mu"] = mu if xdist else mu.transpose(*dims).values
    outcome: Prior | Censored = (
        Censored(unobserved, lower=prior.lower, upper=prior.upper)
        if isinstance(prior, Censored)
        else unobserved
    )
    return outcome.create_variable(name, xdist=xdist)


def _align_to_dims(
    var: XTensorVariable, dims: tuple[str, ...], model: Model
) -> XTensorVariable:
    """Broadcast ``var`` up to ``dims``, in that order.

    ``pmd.Deterministic`` takes the dims off the graph and only transposes
    them, so a quantity built from parameters that do not carry every model
    dim would be stored without those dims. Sizes come from the model, which
    knows every dim it has registered, coords argument or not.
    """
    unknown = [
        dim for dim in dims if dim not in var.dims and dim not in model.dim_lengths
    ]
    if unknown:
        raise ValueError(
            f"Dims {unknown} are not part of the model coords. Add them at "
            "initialization time or use `model.add_coord`."
        )
    missing = {dim: model.dim_lengths[dim] for dim in dims if dim not in var.dims}
    if missing:
        var = var.expand_dims(missing)
    return var.transpose(*dims)


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
        prior predictive sampling is possible. Axis labels are read from
        the data itself (an ``xr.DataArray``) or from the model (a
        ``pm.Data`` registered with dims); anything else, such as a plain
        array or a ``pm.Data`` without dims, is labelled positionally in
        ``(T, ...)`` order with the extra dims following their first
        appearance across the ``p``, ``q``, ``m`` and ``likelihood``
        priors, in that order.
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
        # Declaration order, not set order: `combined_dims` labels the axes of
        # `observed` positionally, so an order that varies between processes
        # would silently mislabel the data.
        declared_dims = (
            *(priors["p"].dims or ()),
            *(priors["q"].dims or ()),
            *(priors["m"].dims or ()),
            *(getattr(priors["likelihood"], "dims", ()) or ()),
        )
        combined_dims = (
            "T",
            *(dim for dim in dict.fromkeys(declared_dims) if dim != "T"),
        )

        time = pmd.as_xtensor(t, dims=("T",))
        m = _create_dim_variable(priors["m"], "m")
        p = _create_dim_variable(priors["p"], "p")
        q = _create_dim_variable(priors["q"], "q")

        def deterministic(name: str, value: XTensorVariable) -> XTensorVariable:
            return pmd.Deterministic(name, _align_to_dims(value, combined_dims, model))

        adopters = deterministic("adopters", m * f(p, q, time))
        deterministic("innovators", m * p * (1 - F(p, q, time)))
        deterministic("imitators", m * q * F(p, q, time) * (1 - F(p, q, time)))

        # `peak` carries only the parameter dims, but in `combined_dims` order
        # so it agrees with the deterministics above.
        peak = (pmd.math.log(q) - pmd.math.log(p)) / (p + q)
        pmd.Deterministic(
            "peak", peak.transpose(*(dim for dim in combined_dims if dim in peak.dims))
        )

        priors["likelihood"].dims = combined_dims
        if observed is None:
            observed_xt = None
        else:
            # The data knows its own axis labels, which `combined_dims` need
            # not match: an `xr.DataArray` carries them, a registered
            # `pm.Data` has them on the model.
            observed_dims = getattr(observed, "dims", None) or (
                model.named_vars_to_dims.get(
                    getattr(observed, "name", None), combined_dims
                )
            )
            observed_xt = pmd.as_xtensor(observed, dims=tuple(observed_dims))

        _create_likelihood_variable(
            priors["likelihood"],
            "y",
            mu=adopters,
            observed=observed_xt,
            dims=combined_dims,
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

        azp.plot_forest(idata.posterior["peak"], combined=True)
    """

    _model_type = "BassModel"
    version = __version__

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        super().__init__(model_config=model_config, sampler_config=sampler_config)
        # Restore Prior objects from the dicts produced by the JSON
        # round-trip in save/load
        self.model_config = parse_model_config(self.model_config)
        self.data: xr.Dataset | None = None

    @property
    def default_model_config(self) -> dict:
        """Default model configuration with weakly informative priors.

        ``m`` is the market potential, a headcount, so its prior is restricted to
        positive values. A prior straddling zero puts the Poisson mean at zero at
        ``model.initial_point()``, making the likelihood ``-inf`` there; ``fit(
        method="map")`` then fails outright, while ``"mcmc"`` only survives it
        because ``jitter+adapt_diag`` moves off the starting point.

        The default ``m`` prior here is a placeholder: because ``m`` is a headcount
        whose scale is entirely dataset-dependent, :meth:`build_model` builds the graph
        with ``HalfNormal(sigma=2 * observed.sum())`` instead whenever the user has not
        overridden it. The rescale is applied per build and does not modify
        ``model_config``, so every fit is scaled to the data it is looking at. Pass an
        ``m`` prior that differs from the default in ``model_config`` to opt out; a
        prior identical to the default is indistinguishable from not passing one.
        """
        return {
            "m": Prior("HalfNormal", sigma=10),
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
            old_value = self.model["y_obs"].get_value()
            dims = self.model.named_vars_to_dims["y_obs"]
            new_shape = tuple(
                len(new_t) if d == "T" else size
                for d, size in zip(dims, old_value.shape, strict=True)
            )
            set_data["y_obs"] = np.zeros(new_shape, dtype=old_value.dtype)
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
            self.idata.update(post_pred)

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

        # `m` is the market potential -- the total number of eventual adopters -- so a
        # fixed-scale default prior is wrong for almost every dataset. When the user has
        # not overridden the default `m` prior, rescale it to the data: the observed
        # cumulative adoptions are a lower bound on `m`, so twice that keeps the prior
        # weakly informative at the right order of magnitude.
        #
        # The resolved prior is local to this build and is deliberately *not* written
        # back into `self.model_config`: doing so would make the check below read a
        # value it had itself produced, so a second `fit` on a different dataset would
        # keep the first dataset's scale. Leaving the config untouched also keeps `id`
        # identical either side of a save/load round trip, since `build_model` recomputes
        # the same sigma from `fit_data`.
        priors = dict(self.model_config)
        if observed is not None and priors["m"] == self.default_model_config["m"]:
            total_adopters = max(float(observed.sum()), 1.0)
            priors["m"] = Prior("HalfNormal", sigma=2 * total_adopters)

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
                priors=cast(BassPriors, priors),
                coords=coords,
                model=self.model,
            )

    def _prepare_fit(
        self,
        data: xr.Dataset | pd.DataFrame | pd.Series | np.ndarray | None = None,
    ) -> None:
        """Normalize the input data and rebuild the model against it.

        The Bass model bakes the time grid into the graph, so the model is rebuilt on
        every fit rather than reused.
        """
        self.data = to_bass_dataset(data) if data is not None else self.data
        if self.data is None:
            raise ValueError("Data must be provided to fit the Bass model.")
        self.build_model(self.data)

    def fit(  # type: ignore[override]
        self,
        data: xr.Dataset | pd.DataFrame | pd.Series | np.ndarray | None = None,
        *,
        method: SamplingMethod = "mcmc",
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> xr.DataTree:
        """Fit the Bass diffusion model.

        Thin wrapper around :meth:`ModelFitter.fit`; see there for the full parameter
        reference.

        Parameters
        ----------
        data : xr.Dataset, pd.DataFrame, pd.Series, np.ndarray
            Adoption counts over time. See :func:`to_bass_dataset` for formats.
        method : str
            Method used to fit the model. One of ``"mcmc"``, ``"map"``, ``"demz"``,
            ``"advi"`` or ``"fullrank_advi"``.
        progressbar : bool, optional
            Whether to show the progress bar. Defaults to ``True``.
        random_seed : optional
            Random seed for reproducibility.
        sample_kwargs : dict, optional
            Only used by the variational methods; forwarded to ``Approximation.sample``.
        **kwargs
            Additional arguments forwarded to the underlying PyMC routine.

        Returns
        -------
        xarray.DataTree
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
            azp.plot_trace(idata, var_names=["m", "p", "q"])

            # Forest plots of peak adoption time
            azp.plot_forest(idata.posterior["peak"], combined=True)

        For posterior predictive sampling with new time points:

        .. code-block:: python

            pp = model.sample_posterior_predictive(X=new_data)
        """
        return super().fit(
            data=data,
            method=method,
            progressbar=progressbar,
            random_seed=random_seed,
            sample_kwargs=sample_kwargs,
            **kwargs,
        )

    def plot_adoption_curve(
        self, **kwargs: Any
    ) -> tuple[plt.Figure, npt.NDArray[Axes]]:
        """Plot the posterior adoption curve with the observed data.

        See :func:`pymc_marketing.bass.plotting.plot_adoption_curve` for
        the parameters.

        Returns
        -------
        tuple[Figure, ndarray of Axes]
            Figure and the axes.
        """
        return plotting.plot_adoption_curve(self, **kwargs)

    def plot_cumulative(self, **kwargs: Any) -> tuple[plt.Figure, npt.NDArray[Axes]]:
        """Plot the cumulative adoption S-curve with the observed data.

        See :func:`pymc_marketing.bass.plotting.plot_cumulative` for
        the parameters.

        Returns
        -------
        tuple[Figure, ndarray of Axes]
            Figure and the axes.
        """
        return plotting.plot_cumulative(self, **kwargs)

    def plot_decomposition(self, **kwargs: Any) -> tuple[plt.Figure, npt.NDArray[Axes]]:
        """Plot the adoption decomposition into innovators and imitators.

        Per-period innovators and imitators go on the left y-axis and
        cumulative adoption on a twin right y-axis.

        See :func:`pymc_marketing.bass.plotting.plot_decomposition` for
        the parameters.

        Returns
        -------
        tuple[Figure, ndarray of Axes]
            Figure and the primary (left) axes.
        """
        return plotting.plot_decomposition(self, **kwargs)

    def plot_peak(self, **kwargs: Any) -> tuple[plt.Figure, npt.NDArray[Axes]]:
        """Plot the posterior distribution of the peak adoption time.

        See :func:`pymc_marketing.bass.plotting.plot_peak` for
        the parameters.

        Returns
        -------
        tuple[Figure, ndarray of Axes]
            Figure and the axes.
        """
        return plotting.plot_peak(self, **kwargs)

    def build_from_idata(self, idata: xr.DataTree) -> None:
        """Rebuild the model from a ``DataTree`` object.

        Used internally by :meth:`ModelBuilder.load`.
        """
        self.idata = idata
        self.build_model()
