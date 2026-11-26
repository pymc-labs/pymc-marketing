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
"""Latent state-space dynamics for SCMs, integrated out by a Kalman filter.

A :class:`LatentStateSpace` wraps a ``pymc_extras.statespace`` model, such as
``BayesianVARMAX``, as a term of an SCM. Observed equations use its observed states
like any other term, for example
``Equation(observed="sales", mu=media + latent["baseline"])``.

Fitting an SCM integrates the states out: every observed
equation whose mean is one observed state plus other terms, with a Normal
likelihood, becomes a row of the model's observation equation. Their joint
likelihood is the prediction-error decomposition: given the past, each date's
observations are multivariate normal, with the mean and covariance predicted by a
Kalman filter from ``pymc_extras.statespace``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pymc_extras.prior import Prior
from pymc_extras.statespace.filters import StandardFilter
from pymc_extras.statespace.filters.distributions import LinearGaussianStateSpace
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_DIM,
    MATRIX_DIMS,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
)
from pytensor.graph.replace import graph_replace

from pymc_marketing.names.mmm import (
    LATENT_EQUATION_SUFFIX,
    LATENT_OBSERVATIONS_SUFFIX,
    Dim,
)
from pymc_marketing.terms import ModelTerm, Product, Sum
from pymc_marketing.terms._graph import (
    BuildContext,
    Equation,
    GraphTerm,
    walk,
)

# pymc-extras' short names of the system matrices, for time-varying ones
_SHORT_NAMES = {
    "state_intercept": "c",
    "obs_intercept": "d",
    "transition": "T",
    "design": "Z",
    "selection": "R",
    "obs_cov": "H",
    "state_cov": "Q",
}


class LatentStateSpace(GraphTerm):
    """Latent states of a ``pymc_extras.statespace`` model, used inside an SCM.

    The statespace model supplies the system matrices; the SCM supplies its
    parameters and exogenous data. Observed equations use its observed states as
    components, for example ``latent["baseline"]``. In training,
    fitting the SCM always integrates the states out with a Kalman
    filter. Prior predictive sampling instead draws the states forward from the
    prior and each equation's observations around them. The model's own
    observation noise is added to that of the equations, so build it without
    measurement error unless both are wanted.

    The model's dimensions and the variables created here are prefixed with
    ``name``, so several latents can share an SCM: the parameter ``ar_params`` of
    a latent named ``"funnel"`` is the variable ``funnel_ar_params`` with dims
    ``("funnel_observed_state", "funnel_lag_ar", "funnel_observed_state_aux")``.

    Parameters
    ----------
    name : str
        Prefix of the term's variables and dimensions.
    model : type
        A ``pymc_extras.statespace`` model class, e.g. ``BayesianVARMAX``.
    model_kwargs : mapping, optional
        Arguments for ``model``. The class and its arguments are what ``SCM.save``
        stores, so they must be JSON-like values.
    priors : mapping of str to Prior, array-like or callable
        One entry per model parameter (``param_names`` of the built model). A
        ``Prior`` uses the model's dims for that parameter unless it declares
        some of them, named as in the model (without the prefix); labeled
        ``DataArray`` parameters of the ``Prior`` use the same names. An array is
        a fixed value. A callable receives ``make(suffix, prior, dims=(),
        coords=None)``, which creates the variable ``{name}_{suffix}`` and returns
        it as a tensor in the order of ``dims``, and returns the parameter
        assembled from such variables, for example a covariance from per-state
        scales. To save the model, register the callable with
        ``pymc_extras.prior.register_tensor_transform``.
    exog : mapping of str to term, optional
        One ``date``-indexed term per data name of the model. A term with another
        dimension is matched to the model's exogenous names by label.

    Notes
    -----
    The marginalized likelihood is one term per date, each date's observations
    given the past. Pointwise log-likelihoods, and LOO on them, are therefore
    one-step-ahead and not comparable with models that sample their latents;
    use leave-future-out validation instead.

    Examples
    --------
    .. code-block:: python

        import pymc_extras.statespace as pmss

        funnel = LatentStateSpace(
            "funnel",
            pmss.BayesianVARMAX,
            model_kwargs={
                "order": (1, 0),
                "endog_names": ["baseline", "awareness"],
                "exog_state_names": ["brand_spend"],
                "trend": "c",
            },
            priors={...},
            exog={"exogenous_data": Data("brand_spend")},
        )
        sales = Equation(observed="sales", mu=media + funnel["baseline"], ...)
        scm = SCM(sales, ...)
    """

    def __init__(
        self,
        name: str,
        model: type,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
        priors: Mapping[str, Prior | np.ndarray | float | Callable[..., Any]],
        exog: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.model_kwargs = dict(model_kwargs or {})
        self.priors = dict(priors)
        self.exog = dict(exog or {})
        self._statespace = model(**{"verbose": False, **self.model_kwargs})
        ss = self._statespace
        missing = set(ss.param_names) - set(self.priors)
        extra = set(self.priors) - set(ss.param_names)
        if missing or extra:
            raise ValueError(
                f"{name!r} needs one prior per model parameter; missing {sorted(missing)}, unknown {sorted(extra)}."
            )
        if set(self.exog) != set(ss.data_names):
            raise ValueError(
                f"{name!r} needs one term per model data name {sorted(ss.data_names)}, got {sorted(self.exog)}."
            )

    @property
    def components(self) -> tuple[str, ...]:
        """Names usable as ``latent[name]``: the model's observed states."""
        return tuple(self._statespace.observed_states)

    @property
    def component_dim(self) -> str:
        """Dimension of the observed states in the simulated path."""
        return self._dim(OBS_STATE_DIM)

    def __getitem__(self, component: str) -> "LatentState":
        """Return one observed state of the model as a term."""
        if component not in self.components:
            raise KeyError(
                f"{component!r} is not an observed state of {self.name!r}; choose from {list(self.components)}."
            )
        return LatentState(self, component)

    def dependencies(self) -> tuple[Any, ...]:
        """Return the exogenous terms."""
        return tuple(self.exog.values())

    def filter_system(self, context: BuildContext) -> SimpleNamespace:
        """Return the model's system matrices bound to SCM variables.

        pymc-extras builds the matrices from placeholder variables. Each placeholder
        is replaced by an SCM variable: parameters come from ``priors``, exogenous
        data from ``exog``, and the number of time steps from the dataset. Each build
        binds a latent once: either its likelihood or its forward simulation does it.
        """
        ss = self._statespace
        context._add_model_coords(
            {self._dim(dim): list(labels) for dim, labels in ss.coords.items()}
        )
        n_steps = context.ds.sizes[Dim.DATE]
        matrices = list(ss._unpack_statespace_with_placeholders())
        replacements = {
            **{
                placeholder: self._parameter(context, param, placeholder)
                for param, placeholder in ss._name_to_variable.items()
            },
            **{
                placeholder: self._exogenous(context, data_name, placeholder)
                for data_name, placeholder in ss._name_to_data.items()
            },
            **{
                placeholder: pt.constant(n_steps, dtype=placeholder.dtype)
                for placeholder in ss._free_n_timesteps(matrices)
            },
        }
        a0, P0, c, d, T, Z, R, H, Q = graph_replace(
            matrices, replace=replacements, strict=False
        )
        return SimpleNamespace(
            a0=a0,
            P0=P0,
            c=_per_date(c, n_steps),
            d=_per_date(d, n_steps),
            T=T,
            Z=Z,
            R=R,
            H=H,
            Q=Q,
            n_steps=n_steps,
            # The intercepts are always passed per date
            time_varying=tuple(
                set(ss.ssm.time_varying_names) | {"state_intercept", "obs_intercept"}
            ),
        )

    def _parameter(
        self, context: BuildContext, param: str, placeholder: Any
    ) -> pt.TensorVariable:
        """One model parameter, from its entry in ``priors``, in the model's layout."""
        spec = self.priors[param]
        dims = tuple(self._statespace.param_dims.get(param, ()))
        shape = placeholder.type.shape
        if isinstance(spec, Prior):
            variable = self._variable(context, param, spec, dims)
            wanted = self._dims(dims)
            if extra := set(variable.dims) - set(wanted):
                raise ValueError(
                    f"The prior of {param!r} in {self.name!r} has dims {sorted(extra)} "
                    f"the model does not use; it uses {list(dims)}."
                )
            # Reorder to the model's dims and broadcast over the ones not declared
            value = pt.expand_dims(
                variable.transpose(
                    *[dim for dim in wanted if dim in variable.dims]
                ).values,
                [i for i, dim in enumerate(wanted) if dim not in variable.dims],
            )
        elif callable(spec):
            value = pt.as_tensor(spec(partial(self._tensor, context)))
            name = f"{self.name}_{param}"
            if len(dims) == value.ndim and name not in context.model.named_vars:
                context._claim_name(name, self)
                pmd.Deterministic(name, pmd.as_xtensor(value, dims=self._dims(dims)))
        else:
            value = pt.as_tensor(np.asarray(spec, dtype="float64"))
        value = value.astype(placeholder.dtype)
        if isinstance(spec, Prior):
            return pt.broadcast_to(value, shape)
        return pt.specify_shape(value, shape)

    def _exogenous(
        self, context: BuildContext, data_name: str, placeholder: Any
    ) -> pt.TensorVariable:
        """One exogenous data input, from its term in ``exog``, shaped as the model's.

        The inputs are matched to the model's labels for them, not taken by position.
        """
        term = context.build(self.exog[data_name], name=f"{self.name}_{data_name}")
        other = [dim for dim in term.dims if dim != Dim.DATE]
        if Dim.DATE not in term.dims or len(other) > 1:
            raise ValueError(
                f"Exogenous term {data_name!r} of {self.name!r} must be indexed by "
                f"{Dim.DATE!r} and at most one other dimension, got {term.dims}."
            )
        input_dims = self._statespace.data_info[data_name]["dims"][1:]
        wanted = (
            list(self._statespace.coords[input_dims[0]])
            if input_dims and input_dims[0] in self._statespace.coords
            else None
        )
        if other:
            (dim,) = other
            labels = (
                context.ds.coords[dim].values.tolist()
                if dim in context.ds.coords
                else None
            )
            if (
                wanted is None
                or labels is None
                or sorted(map(str, labels)) != sorted(map(str, wanted))
            ):
                raise ValueError(
                    f"The {dim!r} labels of exogenous term {data_name!r} ({labels}) "
                    f"must match the model's inputs {wanted}."
                )
            position = {str(label): i for i, label in enumerate(labels)}
            term = term.isel(
                {dim: np.array([position[str(label)] for label in wanted])}
            )
        elif wanted is not None and len(wanted) != 1:
            raise ValueError(
                f"Exogenous term {data_name!r} of {self.name!r} has one series; the model expects {wanted}."
            )
        values = term.transpose(Dim.DATE, *other).values
        if placeholder.ndim == 1:
            return values.reshape((values.shape[0],)).astype(placeholder.dtype)
        return values.reshape((values.shape[0], -1)).astype(placeholder.dtype)

    def _variable(
        self,
        context: BuildContext,
        suffix: str,
        prior: Prior,
        dims: Sequence[str] = (),
        coords: Mapping[str, Sequence[Any]] | None = None,
    ) -> Any:
        """Create the variable ``{name}_{suffix}`` from ``prior``, with prefixed dims."""
        context._add_model_coords(
            {self._dim(dim): list(labels) for dim, labels in (coords or {}).items()}
        )
        name = f"{self.name}_{suffix}"
        context._claim_name(name, self)
        return context._recipe(self._prefixed(prior, dims)).create_variable(
            name, xdist=True
        )

    def _tensor(
        self,
        context: BuildContext,
        suffix: str,
        prior: Prior,
        dims: Sequence[str] = (),
        coords: Mapping[str, Sequence[Any]] | None = None,
    ) -> pt.TensorVariable:
        """``make`` for prior callables: the variable as a tensor, in the order of ``dims``."""
        variable = self._variable(context, suffix, prior, dims, coords)
        return variable.transpose(*self._dims(dims or prior.dims)).values

    def _prefixed(self, prior: Prior, dims: Sequence[str]) -> Prior:
        """Copy ``prior`` with its own dims, and those of labeled parameters, prefixed."""
        prior = prior.deepcopy()
        for key, value in prior.parameters.items():
            if isinstance(value, Prior):
                prior.parameters[key] = self._prefixed(value, ())
            elif isinstance(value, xr.DataArray):
                prior.parameters[key] = value.rename(
                    dict(zip(value.dims, self._dims(value.dims), strict=True))
                )
        # Set after the parameters, whose dims are checked against these
        prior.dims = self._dims(prior.dims or dims)
        return prior

    def _labeled(self, matrix: pt.TensorVariable, short_name: str) -> Any:
        """Label a system matrix with pymc-extras' dims for it, led by the date when per date."""
        dims = self._dims(MATRIX_DIMS[short_name])
        if matrix.ndim == len(dims) + 1:
            dims = (Dim.DATE, *dims)
        return ptx.as_xtensor(matrix, dims=dims)

    def _dim(self, dim: str) -> str:
        return f"{self.name}_{dim}"

    def _dims(self, dims: Sequence[str]) -> tuple[str, ...]:
        return tuple(self._dim(dim) for dim in dims)

    def _build(self, context: BuildContext) -> Any:
        """Draw the states forward from the prior; used for prior predictive sampling.

        Uses pymc-extras' ``LinearGaussianStateSpace``, which follows the filter's
        timing. The state path is ``{name}_latent`` and the observed states, with the
        model's own noise, ``{name}_observed``.
        """
        if context.prediction:
            # Forecasting starts from the state at the end of the fitted data instead
            raise NotImplementedError(
                f"Forecasting {self.name!r} needs its state at the end of the fitted "
                "data, which is not supported yet."
            )
        system = self.filter_system(context)
        # pymc-extras' LinearGaussianStateSpace names its draws after the latent
        for suffix in ("combined", "latent", "observed"):
            context._claim_name(f"{self.name}_{suffix}", self)
        _, observed = LinearGaussianStateSpace(
            self.name,
            system.a0,
            system.P0,
            system.c,
            system.d,
            system.T,
            system.Z,
            system.R,
            system.H,
            system.Q,
            steps=system.n_steps - 1,
            sequence_names=[_SHORT_NAMES[name] for name in system.time_varying],
            dims=(Dim.DATE, self._dim(ALL_STATE_DIM), self.component_dim),
        )
        return pmd.as_xtensor(observed, dims=(Dim.DATE, self.component_dim))


class LatentState(GraphTerm):
    """One observed state of a :class:`LatentStateSpace`, as a ``date``-indexed term."""

    def __init__(self, latent: LatentStateSpace, state: str) -> None:
        self.latent = latent
        self.state = state

    def dependencies(self) -> tuple[Any, ...]:
        """Return the latent term."""
        return (self.latent,)

    def _build(self, context: BuildContext) -> Any:
        path = context.build(self.latent)
        return path.isel(
            {self.latent.component_dim: self.latent.components.index(self.state)}
        )


@dataclass
class _Observation:
    """An equation read as ``Normal(mu=offset + latent[state], sigma)``."""

    equation: Equation
    state: str
    offset: ModelTerm | Sum | Product | None
    sigma: Prior | float | ModelTerm | Sum | Product


@dataclass
class _KalmanLikelihood:
    """Joint likelihood of the equations observing one marginalized latent.

    Built after the rest of the model: the equations' values are their observations
    (see ``_observed_value``), so every offset can be built first, including ones that
    use another of these equations.
    """

    latent: LatentStateSpace
    observations: list[_Observation] = field(default_factory=list)

    def build(self, context: BuildContext) -> None:
        latent = self.latent
        equation = f"{latent.name}{LATENT_EQUATION_SUFFIX}"
        equation_aux = f"{equation}_aux"
        bindings = [context.binding(obs.equation) for obs in self.observations]
        names = [binding.name for binding in bindings]
        context._add_model_coords({equation: names, equation_aux: names})

        def by_equation(values: list[Any]) -> Any:
            return ptx.concat(
                [value.expand_dims(equation) for value in values], dim=equation
            )

        observed = [context._observations(binding) for binding in bindings]
        data = by_equation(observed)
        offsets = by_equation(
            [
                self._offset(context, obs, name, values)
                for obs, name, values in zip(
                    self.observations, names, observed, strict=True
                )
            ]
        )
        noise = by_equation(
            [
                self._sigma(context, obs, name)
                for obs, name in zip(self.observations, names, strict=True)
            ]
        )

        # Each equation observes the row of the model's observation equation for its
        # state, selected by dim name, plus its own offset and noise
        system = latent.filter_system(context)
        rows = ptx.as_xtensor(
            np.array([latent.components.index(obs.state) for obs in self.observations]),
            dims=(equation,),
        )
        rows_aux = rows.rename({equation: equation_aux})
        observed_state = latent._dim(OBS_STATE_DIM)
        observed_state_aux = latent._dim(OBS_STATE_AUX_DIM)
        d = latent._labeled(system.d, "d").isel({observed_state: rows}) + offsets
        Z = latent._labeled(system.Z, "Z").isel({observed_state: rows})
        H = latent._labeled(system.H, "H").isel(
            {observed_state: rows, observed_state_aux: rows_aux}
        ) + noise**2 * ptx.as_xtensor(pt.eye(len(names)), dims=(equation, equation_aux))

        def plain(value: Any, *dims: str) -> pt.TensorVariable:
            """Put value in the filter's layout: the date first when per date, then dims."""
            return value.transpose(
                *([Dim.DATE] if Dim.DATE in value.dims else []), *dims
            ).values

        predicted = StandardFilter(time_varying_names=system.time_varying).build_graph(
            plain(data, equation),
            system.a0,
            system.P0,
            system.c,
            plain(d, equation),
            system.T,
            plain(Z, equation, latent._dim(ALL_STATE_DIM)),
            system.R,
            plain(H, equation, equation_aux),
            system.Q,
        )
        predicted_mean, predicted_cov = predicted[2], predicted[5]

        name = f"{latent.name}{LATENT_OBSERVATIONS_SUFFIX}"
        context._claim_name(name, latent)
        pmd.MvNormal(
            name,
            mu=pmd.as_xtensor(predicted_mean, dims=(Dim.DATE, equation)),
            cov=pmd.as_xtensor(predicted_cov, dims=(Dim.DATE, equation, equation_aux)),
            core_dims=(equation, equation_aux),
            observed=data,
            dims=(Dim.DATE, equation),
        )

    @staticmethod
    def _offset(
        context: BuildContext, obs: _Observation, name: str, observed: Any
    ) -> Any:
        """Build the equation's other terms, over the equation's dates."""
        if obs.offset is None:
            return ptx.zeros_like(observed)
        offset = context.build(obs.offset, name=f"{name}_offset")
        return offset.broadcast_like(observed)

    @staticmethod
    def _sigma(context: BuildContext, obs: _Observation, name: str) -> Any:
        """Build the equation's noise scale, a scalar."""
        if isinstance(obs.sigma, Prior):
            context._claim_name(f"{name}_sigma", obs.equation)
            sigma = context._recipe(obs.sigma).create_variable(
                f"{name}_sigma", xdist=True
            )
        elif isinstance(obs.sigma, int | float):
            sigma = ptx.as_xtensor(float(obs.sigma), dims=())
        else:
            sigma = context.build(obs.sigma, name=f"{name}_sigma")
        if sigma.dims:
            raise ValueError(
                f"The sigma of {name!r} must be a scalar when its latent is marginalized."
            )
        return sigma


def _per_date(value: pt.TensorVariable, n_steps: int) -> pt.TensorVariable:
    """Repeat a constant intercept over dates; a time-varying one is kept."""
    if value.ndim == 2:
        return value
    return pt.broadcast_to(value, (n_steps, value.shape[0]))


def _observation(equation: Equation, latent: LatentStateSpace) -> _Observation | None:
    """Read ``equation`` as ``Normal(mu=<terms> + latent[state], sigma)``.

    Returns None when the equation does not use the latent, and refuses any other
    use of it, since its states are no longer built.
    """

    def uses_latent(root: Any) -> bool:
        # A nested equation is planned on its own
        return any(
            node is latent
            for node in walk(
                root,
                stop=lambda node: isinstance(node, Equation) and node is not equation,
            )
        )

    if not uses_latent(equation):
        return None
    if equation.observed is None:
        raise ValueError(
            f"Equation {equation.name!r} uses {latent.name!r}, which only observed "
            "equations can do when it is marginalized."
        )

    def is_state(node: Any) -> bool:
        return isinstance(node, LatentState) and node.latent is latent

    if equation.likelihood.transform or not equation.likelihood.centered:
        raise ValueError(
            "Observed equations do not support transformed or noncentered likelihood recipes."
        )
    mu = equation.parameters.get("mu")
    terms = [] if mu is None else list(mu.terms) if isinstance(mu, Sum) else [mu]
    states = [term for term in terms if is_state(term)]
    rest = [term for term in terms if not is_state(term)]
    sigma = equation.likelihood.parameters.get(
        "sigma", equation.parameters.get("sigma")
    )
    # The latent may only enter the mean: its path is never built when fitting
    parameters = set(equation.parameters) | set(equation.likelihood.parameters)
    if (
        equation.likelihood.distribution != "Normal"
        or sigma is None
        or parameters - {"mu", "sigma"}
        or len(states) != 1
        or any(uses_latent(term) for term in rest)
        or uses_latent(sigma)
    ):
        raise ValueError(
            f"Equation {equation.name or equation.observed!r} must be "
            f"Normal(mu=<terms> + {latent.name}[<state>], sigma) to marginalize {latent.name!r}."
        )
    offset = None if not rest else rest[0] if len(rest) == 1 else Sum(terms=rest)
    return _Observation(equation, states[0].state, offset, sigma)


def _observed_value(equation: Equation, context: BuildContext) -> Any:
    """Build an equation that observes a marginalized latent: its value is its data.

    Its likelihood is built later with the latent's other equations, so building the
    equation never depends on them.
    """
    binding = context.register_equation(equation)
    if binding.dims != (Dim.DATE,):
        raise ValueError(
            f"Equation {binding.name!r} must be indexed by {Dim.DATE!r} only."
        )
    return context._observations(binding)


def _plan(
    equations: Sequence[Equation], latents: Sequence[LatentStateSpace]
) -> list[_KalmanLikelihood]:
    """Group the equations observing each marginalized latent into its likelihood."""
    plan: list[_KalmanLikelihood] = []
    seen: set[int] = set()
    for latent in latents:
        likelihood = _KalmanLikelihood(latent)
        for equation in equations:
            observation = _observation(equation, latent)
            if observation is None:
                continue
            if id(equation) in seen:
                raise ValueError(
                    f"Equation {equation.name or equation.observed!r} observes more than one marginalized latent."
                )
            seen.add(id(equation))
            likelihood.observations.append(observation)
        if not likelihood.observations:
            raise ValueError(f"No equation observes {latent.name!r}.")
        plan.append(likelihood)
    return plan


class Marginalization:
    """How an SCM build handles the latent state-space terms its equations reach.

    Fitting integrates every latent out: the equations observing it build to their
    observations, and its joint likelihood is built after the rest of the model.
    Prior predictive sampling instead draws the latents forward when the equations
    reach them, so only the checks apply. The checks run on construction, in both
    cases, so a prior predictive check refuses what fitting would refuse.

    Parameters
    ----------
    roots : sequence of Equation
        The SCM's equations.
    """

    def __init__(self, roots: Sequence[Equation]) -> None:
        nodes = list(walk(roots))
        self._likelihoods = _plan(
            [node for node in nodes if isinstance(node, Equation)],
            [node for node in nodes if isinstance(node, LatentStateSpace)],
        )

    def attach(self, context: BuildContext) -> None:
        """Make the observing equations build to their observations in ``context``."""
        context.equation_builders.update(
            {
                id(obs.equation): _observed_value
                for likelihood in self._likelihoods
                for obs in likelihood.observations
            }
        )

    def build(self, context: BuildContext) -> None:
        """Add each latent's joint likelihood; call after building the equations."""
        for likelihood in self._likelihoods:
            likelihood.build(context)
