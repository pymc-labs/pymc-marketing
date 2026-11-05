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
"""
Specialized priors that behave like the Prior class.

The Prior class has certain design constraints that prevent it from
covering all cases. So this module contains a collection of
priors that do not inherit from the Prior class but have many
of the same methods.
"""

import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import xarray as xr
from numpy.typing import (
    ArrayLike,  # resolves pt.TensorLike's ForwardRef('ArrayLike') for sphinx_autodoc_typehints (#1197)
)
from pymc.distributions.dist_math import check_parameters
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import (
    MuAlreadyExistsError,
    Prior,
    Scaled,
    VariableFactory,
    _param_value_with_dims,
    sample_prior,
)
from pytensor.tensor import TensorVariable, as_tensor
from pytensor.xtensor.type import XTensorVariable, as_xtensor

from pymc_marketing.serialization import serialization


def _is_xarray_dataarray_dict(data: Any) -> bool:
    """Return True for dicts produced by ``xr.DataArray.to_dict()``."""
    if not isinstance(data, dict):
        return False
    required = {"dims", "attrs", "data", "coords"}
    return (
        required.issubset(data.keys())
        and "data_vars" not in data
        and "dist" not in data
        and "special_prior" not in data
        and "lookup_name" not in data
    )


register_deserialization(
    is_type=_is_xarray_dataarray_dict,
    deserialize=xr.DataArray.from_dict,
)


def _serialize_scaled(obj: Scaled) -> dict:
    return {
        "__type__": f"{Scaled.__module__}.{Scaled.__qualname__}",
        "dist": obj.dist.to_dict(),
        "factor": obj.factor,
    }


def _deserialize_scaled(data: dict, context: Any = None) -> Scaled:
    return Scaled(dist=deserialize(data["dist"]), factor=data["factor"])


serialization.register(
    Scaled,
    serializer=_serialize_scaled,
    deserializer=_deserialize_scaled,
)


class MeanAlreadyExistsError(MuAlreadyExistsError):
    """Error for when ``mean`` is already present in a likelihood prior.

    Subclasses :class:`pymc_extras.prior.MuAlreadyExistsError` so existing
    ``except MuAlreadyExistsError`` handlers keep working, but the message
    names ``mean``, the key that actually collides in
    :meth:`LogNormalPrior.create_likelihood_variable`.
    """

    def __init__(self, distribution) -> None:
        super().__init__(distribution)
        self.message = (
            f"The mean parameter (or its alias mu) is already defined in {distribution}"
        )
        self.args = (self.message,)


class SpecialPrior(ABC):
    """A base class for specialized priors."""

    def __init__(self, dims: tuple | None = None, centered: bool = True, **parameters):
        self.dims = dims
        self.centered = centered
        self.parameters = parameters
        self._checks()

    def __eq__(self, other):
        """Check equality based on class, dims, centered, and parameters."""
        if not isinstance(other, self.__class__):
            return False
        if self.dims != other.dims or self.centered != other.centered:
            return False

        # Compare parameters, handling numpy arrays
        if set(self.parameters.keys()) != set(other.parameters.keys()):
            return False

        for key in self.parameters:
            val1 = self.parameters[key]
            val2 = other.parameters[key]

            # Handle numpy arrays
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    return False
            elif isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                return False  # One is array, other isn't
            elif val1 != val2:
                return False

        return True

    def __hash__(self):
        """Compute hash based on class, dims, centered, and parameters."""
        # Convert parameters to a hashable tuple
        param_items = []
        for key in sorted(self.parameters.keys()):
            value = self.parameters[key]
            # Convert numpy arrays to tuples for hashing
            if isinstance(value, np.ndarray):
                value = tuple(value.flat)
            # Convert lists to tuples
            elif isinstance(value, list):
                value = tuple(value)
            # For unhashable types, use their string representation
            try:
                hash(value)
            except TypeError:
                value = str(value)
            param_items.append((key, value))

        return hash(
            (self.__class__.__name__, self.dims, self.centered, tuple(param_items))
        )

    @abstractmethod
    def _checks(self) -> None:  # pragma: no cover
        """Check that the parameters are correct."""
        pass

    @abstractmethod
    def create_variable(self, name: str) -> TensorVariable:  # pragma: no cover
        """Create a variable from the prior distribution."""
        pass

    def _create_parameter(self, param, value, name, xdist: bool = False):
        if not hasattr(value, "create_variable"):
            if xdist:
                return _param_value_with_dims(param, value, dims=self.dims)
            else:
                return value

        child_name = f"{name}_{param}"
        return value.create_variable(child_name, xdist=xdist)

    def to_dict(self):
        """Convert the SpecialPrior to a dictionary."""
        class_name = self.__class__.__name__
        data = {
            "special_prior": class_name,
        }
        if self.parameters:

            def handle_value(value):
                if isinstance(value, Prior):
                    return value.to_dict()

                if isinstance(value, pt.TensorVariable):
                    value = value.eval()

                if isinstance(value, np.ndarray):
                    return value.tolist()

                if isinstance(value, xr.DataArray):
                    return value.to_dict()

                if hasattr(value, "to_dict"):
                    return value.to_dict()

                return value

            data["kwargs"] = {
                param: handle_value(value) for param, value in self.parameters.items()
            }
        if not self.centered:
            data["centered"] = False

        if self.dims:
            data["dims"] = self.dims

        return data

    @classmethod
    def from_dict(cls, data) -> "SpecialPrior":
        """Create a SpecialPrior prior from a dictionary."""
        if not isinstance(data, dict):
            msg = (
                "Must be a dictionary representation of a prior distribution. "
                f"Not of type: {type(data)}"
            )
            raise ValueError(msg)

        data = {k: v for k, v in data.items() if k != "__type__"}

        # Extract special keys
        centered = data.get("centered", True)
        dims = data.get("dims")
        # Convert dims to tuple if it's a list (e.g., from YAML)
        if isinstance(dims, list):
            dims = tuple(dims)

        # Check if parameters are in kwargs or at top level
        if "kwargs" in data:
            # Parameters are in kwargs subdictionary
            kwargs = data.get("kwargs", {})
        else:
            # Parameters are at top level - extract everything except special keys
            special_keys = {"special_prior", "centered", "dims"}
            kwargs = {k: v for k, v in data.items() if k not in special_keys}

        def handle_value(value):
            if isinstance(value, dict):
                return deserialize(value)

            if isinstance(value, list):
                return np.array(value)

            return value

        kwargs = {param: handle_value(value) for param, value in kwargs.items()}

        return cls(dims=dims, centered=centered, **kwargs)

    def sample_prior(
        self,
        coords=None,
        name: str = "variable",
        **sample_prior_predictive_kwargs,
    ) -> xr.Dataset:
        """Sample from the prior distribution."""
        return sample_prior(
            factory=self,
            coords=coords,
            name=name,
            xdist=True,
            **sample_prior_predictive_kwargs,
        )


@serialization.register
class LogNormalPrior(SpecialPrior):
    r"""Lognormal prior parameterized by positive-scale mean and std.

    This prior differs from the standard ``LogNormal`` distribution,
    which takes log-scale parameters (``mu_log``, ``sigma_log``).
    Instead, it is parameterized directly in terms of the mean and
    standard deviation (``mean``, ``std``) on the positive scale, making it more intuitive
    and suitable for hierarchical modeling.

    To achieve this, the lognormal parameters are computed internally
    from the positive-domain parameters:

    .. math::

        \mu_{\text{log}} &= \ln \left( \frac{\text{mean}^2}{\sqrt{\text{mean}^2 + \text{std}^2}} \right) \\
        \sigma_{\text{log}} &= \sqrt{ \ln \left( 1 + \frac{\text{std}^2}{\text{mean}^2} \right) }

    where :math:`\text{mean} > 0` and :math:`\text{std} > 0`.

    The prior is then defined as:

    .. math::

        \phi \sim \text{LogNormal}(\mu_{\text{log}}, \sigma_{\text{log}})

    This construction ensures that the resulting random variable
    has approximately the intended mean and variance on the positive scale,
    even when :math:`\text{mean}` and :math:`\text{std}` are themselves random variables.

    Parameters
    ----------
    mean : Prior, float, int, array-like, optional
        The mean of the distribution on the positive scale. It can be omitted
        only when the instance is used as a likelihood, in which case the mean
        is supplied later as ``mu`` by :meth:`create_likelihood_variable`.
    std : Prior, float, int, array-like
        The standard deviation of the distribution on the positive scale.
    dims : tuple[str, ...], optional
        The dimensions of the distribution, by default None.
    centered : bool, optional
        Whether to use the centered parameterization, by default True.
        Ignored when the instance is used as a likelihood, since an observed
        variable has no non-centered form.

    Examples
    --------
    Build a non-centered hierarchical model where information is shared across groups:

    .. code-block:: python

        from pymc_marketing.special_priors import LogNormalPrior

        prior = LogNormalPrior(
            mean=Prior("Gamma", mu=1.0, sigma=1.0),
            std=Prior("HalfNormal", sigma=1.0),
            dims=("geo",),
            centered=False,
        )

    Use it as an MMM likelihood under ``link="identity"`` by configuring only
    ``std``. The linear predictor ``mu`` becomes the response-scale mean, so
    unlike a plain ``LogNormal`` likelihood the additive decomposition stays
    in the units of the target. Both the target and the fitted ``mu`` must be
    strictly positive; during sampling a non-positive ``mu`` is rejected with
    ``-inf`` log-probability rather than being silently folded to ``|mu|``.
    This guard protects only log-probability evaluation: forward sampling is
    rewritten so that a non-positive ``mu`` produces draws of exactly
    ``exp(-inf) = 0``, with no error (see
    :meth:`create_likelihood_variable`); the MMM warns when prior or
    posterior predictive draws show this signature. The MMM's default
    identity-link intercept prior is centered at zero, so roughly a third of
    prior-predictive draws hit this path out of the box; pair this likelihood
    with a positive intercept prior (for example
    ``Prior("HalfNormal", sigma=2)`` or a ``Normal`` centered well above
    zero) so the prior puts ``mu`` on the positive side.

    This parameterization also changes the noise model. Holding ``std``
    constant across dates makes ``sigma_log`` shrink as ``mu`` grows, so the
    likelihood models (approximately) constant additive noise of size
    ``std`` in the units of the target. A plain ``LogNormal`` likelihood
    with fixed ``sigma`` instead models constant relative noise, with a
    standard deviation proportional to the mean. Note the likelihood scale
    variable is named ``y_std``, not ``y_sigma`` as with the default
    ``Normal`` likelihood.

    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import LogNormalPrior

        likelihood = LogNormalPrior(
            std=Prior("HalfNormal", sigma=0.5, dims=("country",)),
            dims=("date", "country"),
        )
        mmm = MMM(..., model_config={"likelihood": likelihood})

    References
    ----------
    - D. Saunders, `A positive constrained non-centered prior that sparks joy <https://daniel-saunders-phil.github.io/imagination_machine/posts/a-positive-constrained-non-centered-prior-that-sparks-joy/>`_.
    - Wikipedia, `Log-normal distribution — Definitions <https://en.wikipedia.org/wiki/Log-normal_distribution#Definitions>`_.
    """

    def __init__(self, dims: tuple | None = None, centered: bool = True, **parameters):
        # Accept aliases mu->mean and sigma->std for convenience/compatibility
        if "mean" not in parameters and "mu" in parameters:
            parameters["mean"] = parameters.pop("mu")
        if "std" not in parameters and "sigma" in parameters:
            parameters["std"] = parameters.pop("sigma")

        super().__init__(dims=dims, centered=centered, **parameters)

    def _checks(self) -> None:
        self._parameters_are_correct_set()

    def _parameters_are_correct_set(self) -> None:
        # Only allow exactly these keys after alias normalization. A std-only
        # instance is the likelihood configuration: the mean arrives later as
        # ``mu`` in ``create_likelihood_variable``.
        if set(self.parameters.keys()) not in ({"mean", "std"}, {"std"}):
            raise ValueError(
                "Parameters must be mean and std, or std alone for likelihood use"
            )

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create a variable from the prior distribution."""
        if not xdist:
            raise NotImplementedError(f"{self!r} only supports xdist=True")
        if "mean" not in self.parameters:
            raise ValueError(
                "LogNormalPrior with only 'std' can only be used as a likelihood "
                "via create_likelihood_variable. Provide 'mean' to use it as a prior."
            )

        parameters = {
            param: as_xtensor(self._create_parameter(param, value, name, xdist=True))
            for param, value in self.parameters.items()
        }
        mu_log = pmd.math.log(
            parameters["mean"] ** 2
            / pmd.math.sqrt(parameters["mean"] ** 2 + parameters["std"] ** 2)
        )
        sigma_log = pmd.math.sqrt(
            pmd.math.log1p(parameters["std"] ** 2 / parameters["mean"] ** 2)
        )

        if self.centered:
            log_phi = pmd.Normal(
                name + "_log", mu=mu_log, sigma=sigma_log, dims=self.dims
            )

        else:
            log_phi_z = pmd.Normal(
                name + "_log" + "_offset", mu=0, sigma=1, dims=self.dims
            )
            log_phi = mu_log + log_phi_z * sigma_log

        phi = pmd.math.exp(log_phi)
        phi = pmd.Deterministic(name, phi)

        return phi

    def validate_observed(self, y: ArrayLike) -> None:
        """Raise if observed data lies outside the LogNormal support.

        A LogNormal observed variable has support on the strictly positive
        reals, so a single zero, negative, or non-finite observation drives
        the whole likelihood to ``-inf`` and the model cannot sample. Callers
        that preprocess the target (for example the MMM, which clamps
        NaN/inf scaled targets to ``0.0``) should invoke this before building
        the model so the failure surfaces as an informative error instead.

        Parameters
        ----------
        y : ArrayLike
            Observed data intended for :meth:`create_likelihood_variable`.

        Raises
        ------
        ValueError
            If ``y`` contains non-positive or non-finite values.
        """
        y_arr = np.asarray(y)
        non_finite = np.count_nonzero(~np.isfinite(y_arr))
        non_positive = np.count_nonzero(y_arr <= 0)
        if non_finite or non_positive:
            raise ValueError(
                "A LogNormalPrior likelihood requires strictly positive, finite "
                f"observed values, but the target contains {non_positive} "
                f"non-positive and {non_finite} non-finite entries. Any such "
                "entry makes the log-likelihood -inf, so the model cannot "
                "sample. Remove or impute these values, or use a likelihood "
                "whose support includes them."
            )

    def create_likelihood_variable(
        self,
        name: str,
        mu: XTensorVariable,
        observed: Any,
        xdist: bool = False,
    ) -> XTensorVariable:
        """Create an observed LogNormal variable whose response-scale mean is ``mu``.

        The instance must be configured with only ``std``; ``mu`` takes the
        role of ``mean`` and the pair is converted to the log-scale
        parameters of the observed ``LogNormal``, so that ``E[y] = mu``.

        The conversion is only valid for ``mu > 0`` and ``std > 0``, and the
        positivity guard protects only log-probability evaluation. During
        sampling a non-positive ``mu`` or ``std`` yields ``-inf``
        log-probability (surfacing as a generic ``SamplingError`` when the
        starting point is invalid), and a direct evaluation of the model
        logp raises a ``ParameterValueError`` naming this requirement. The
        conversion is evaluated on a positive stand-in wherever ``mu <= 0``,
        so the logp is ``-inf`` rather than NaN even where
        ``(std / mu) ** 2`` would overflow; the ``-inf`` log-mean still
        leaves the gradient there non-finite, which NUTS treats as a
        divergence, so the rejection is safe but not clean. The guard is a
        ``pymc.check_parameters`` check and exists only under the default
        ``pm.Model(check_bounds=True)``; with ``check_bounds=False`` a
        non-positive ``mu`` is evaluated as if ``mu = 1`` and produces a
        finite, wrong logp, so do not disable bound checks with this
        likelihood. Forward sampling is not protected at all:
        pymc rewrites the guard in every compiled function, so a
        forward-sampling graph turns a non-positive ``mu`` into a ``-inf``
        log-mean and the draw becomes exactly ``exp(-inf) = 0``, and a
        non-positive ``std`` draw makes the log-scale sigma NaN so the draw
        is NaN, with no error or warning at this level. A genuine LogNormal
        draw is finite and never exactly zero except through float underflow
        of an extremely negative log-scale draw, so exact zeros almost always
        indicate a non-positive ``mu`` and non-finite draws a non-positive
        ``std``. Prior draws, out-of-sample predictions, or counterfactuals
        that push ``mu`` below zero, or a ``std`` prior with mass below zero,
        therefore return zeros or NaNs that look like real predictions;
        :meth:`~pymc_marketing.mmm.mmm.MMM.sample_posterior_predictive` and
        :meth:`~pymc_marketing.mmm.mmm.MMM.sample_prior_predictive`
        detect and warn about both. The sign is never silently
        folded to ``|mu|``.

        The ``centered`` flag is ignored: an observed variable has no
        non-centered parameterization.

        Parameters
        ----------
        name : str
            Name of the observed variable. A ``Prior`` std creates its own
            variable named ``f"{name}_std"``.
        mu : XTensorVariable
            Response-scale mean, e.g. the linear predictor of an MMM under
            ``link="identity"``. Must be positive.
        observed : XTensorLike
            Observed data, strictly positive.
        xdist : bool, optional
            Only ``True`` is supported, matching :meth:`create_variable`.

        Returns
        -------
        XTensorVariable
            The observed variable.

        Raises
        ------
        NotImplementedError
            If ``xdist`` is False.
        MeanAlreadyExistsError
            If the instance already defines ``mean`` (or its alias ``mu``).
            Subclass of ``MuAlreadyExistsError``.
        """
        if not xdist:
            raise NotImplementedError(f"{self!r} only supports xdist=True")
        if "mean" in self.parameters:
            raise MeanAlreadyExistsError(self)

        new = copy.deepcopy(self)
        new.parameters["mean"] = mu

        parameters = {
            param: as_xtensor(new._create_parameter(param, value, name, xdist=True))
            for param, value in new.parameters.items()
        }
        mean = parameters["mean"]
        std = parameters["std"]

        # Evaluate the conversion on a positive stand-in wherever mean <= 0 so
        # the untaken branch stays finite and the guard below is the single
        # source of -inf (under check_bounds=True; without it the stand-in
        # yields a finite, wrong logp); otherwise `log(mean)` there is NaN and
        # (std / mean) ** 2 can overflow, turning the logp itself into NaN.
        # Exact for mean > 0; never folds to |mean| like the squared form
        # log(mean**2 / sqrt(mean**2 + std**2)) would.
        mean_safe = pmd.math.switch(mean > 0, mean, 1.0)
        log_mean = pmd.math.log(mean_safe)
        # log1p((std / mean) ** 2) written as softplus(2 * (log std - log mean)),
        # which stays finite where the ratio itself overflows (|mean| below
        # ~1e-154 with std ~ 1 would give inf, and a NaN logp).
        var_ratio = pmd.math.softplus(2.0 * (pmd.math.log(std) - log_mean))
        mu_log = log_mean - 0.5 * var_ratio
        sigma_log = pmd.math.sqrt(var_ratio)

        # check_parameters/CheckParameterValue is tensor-only, so unwrap the
        # xtensor expressions and re-wrap the checked result.
        mu_log_tensor = check_parameters(
            as_tensor(mu_log, allow_xtensor_conversion=True),
            as_tensor((mean > 0) & (std > 0), allow_xtensor_conversion=True),
            msg="LogNormalPrior likelihood requires mu > 0 and std > 0. The "
            "response-scale mean and standard deviation of a LogNormal must "
            "be positive; the model produced a non-positive value.",
            can_be_replaced_by_ninf=True,
        )
        mu_log = as_xtensor(mu_log_tensor, dims=mu_log.type.dims)

        return pmd.LogNormal(
            name, mu=mu_log, sigma=sigma_log, observed=observed, dims=self.dims
        )


def _is_LogNormalPrior_type(data: dict) -> bool:
    if "special_prior" in data:
        return data["special_prior"] == "LogNormalPrior"
    else:
        return False


register_deserialization(
    is_type=_is_LogNormalPrior_type,
    deserialize=LogNormalPrior.from_dict,
)


@serialization.register
class LaplacePrior(SpecialPrior):
    """A Laplace prior parameterized by a location and a scale parameter.

    Unlike the standard Laplace distribution available through the Prior class,
    this distribution can optionally be centered or non-centered. A non-centered parameterization
    can sometimes eliminate sampling problems in hierarchical models.

    Parameters
    ----------
    mu : Prior, float, int, array-like
        The location parameter of the distribution.
    b : Prior, float, int, array-like
        The scale parameter of the distribution.
    dims : tuple[str, ...], optional
        The dimensions of the distribution, by default None.
    centered : bool, optional
        Whether to use the centered parameterization, by default True.

    References
    ----------
    - A.C. Jones, `Scale mixtures of unbounded continuous distributions <https://andrewcharlesjones.github.io/journal/scale-mixtures.html>`_.
    - Stan Documentation, `Unbounded continuous distributions <https://mc-stan.org/docs/functions-reference/unbounded_continuous_distributions.html#double-exponential-laplace-distribution>`_.
    """

    def _checks(self) -> None:
        self._parameters_are_correct_set()

    def _parameters_are_correct_set(self) -> None:
        # Only allow exactly these keys after alias normalization
        if set(self.parameters.keys()) != {"mu", "b"}:
            raise ValueError("Parameters must be mu and b")

    def create_variable(self, name: str, xdist: bool = False) -> TensorVariable:
        """Create a variable from the prior distribution."""
        if not xdist:
            raise NotImplementedError(f"{self!r} only supports xdist=True")

        parameters = {
            param: self._create_parameter(param, value, name, xdist=True)
            for param, value in self.parameters.items()
        }
        if self.centered:
            phi = pmd.Laplace(
                name, mu=parameters["mu"], b=parameters["b"], dims=self.dims
            )

        else:
            sigma = pmd.Exponential(name + "_sigma", scale=2 * parameters["b"] ** 2)
            phi = pmd.Normal(name + "_offset", mu=0, sigma=1, dims=self.dims)
            phi = pmd.Deterministic(
                name,
                phi * pmd.math.sqrt(sigma) + parameters["mu"],
            )

        return phi


def _is_LaplacePrior_type(data: dict) -> bool:
    if "special_prior" in data:
        return data["special_prior"] == "LaplacePrior"
    else:
        return False


register_deserialization(
    is_type=_is_LaplacePrior_type,
    deserialize=LaplacePrior.from_dict,
)


@serialization.register
class MaskedPrior:
    """Create variables from a prior over only the active entries of a boolean mask.

    Parameters
    ----------
    prior : Prior
        Base prior whose variable is defined over `prior.dims`. Internally, the
        variable is created only for the active entries given by `mask` and
        then expanded back to the full shape with zeros at inactive positions.
    mask : xarray.DataArray
        Boolean array with the same dims and shape as `prior.dims` marking active
        (True) and inactive (False) entries.
    active_dim : str, optional
        Name of the coordinate indexing the active subset. If not provided, a
        name is generated as ``"non_null_dims:<dim1>_<dim2>_..."``. If an existing
        coordinate with the same name has a different length, a suffix with the
        active length is appended.

    Examples
    --------
    Simple 1D masking.

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {"country": ["Venezuela", "Colombia"]}
        mask = xr.DataArray(
            [True, False],
            dims=["country"],
            coords={"country": coords["country"]},
        )
        intercept = Prior("Normal", mu=0, sigma=10, dims=("country",))
        with pm.Model(coords=coords):
            masked = MaskedPrior(intercept, mask)
            intercept_full = masked.create_variable("intercept")

    Nested parameter priors with dims remapped to the active subset.

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {"country": ["Venezuela", "Colombia"]}
        mask = xr.DataArray(
            [True, False],
            dims=["country"],
            coords={"country": coords["country"]},
        )
        intercept = Prior(
            "Normal",
            mu=Prior("HalfNormal", sigma=1, dims=("country",)),
            sigma=10,
            dims=("country",),
        )
        with pm.Model(coords=coords):
            masked = MaskedPrior(intercept, mask)
            intercept_full = masked.create_variable("intercept")

    All entries masked (returns deterministic zeros with original dims).

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {"country": ["Venezuela", "Colombia"]}
        mask = xr.DataArray(
            [False, False],
            dims=["country"],
            coords={"country": coords["country"]},
        )
        prior = Prior("Normal", mu=0, sigma=10, dims=("country",))
        with pm.Model(coords=coords):
            masked = MaskedPrior(prior, mask)
            zeros = masked.create_variable("intercept")

    Apply over a saturation function priors:

    .. code-block:: python

        from pymc_marketing.mmm import LogisticSaturation
        from pymc_marketing.special_priors import MaskedPrior

        coords = {
            "country": ["Colombia", "Venezuela"],
            "channel": ["x1", "x2", "x3", "x4"],
        }

        mask_excluded_x4_colombia = xr.DataArray(
            [[True, False, True, False], [True, True, True, True]],
            dims=["country", "channel"],
            coords=coords,
        )

        saturation = LogisticSaturation(
            priors={
                "lam": MaskedPrior(
                    Prior(
                        "Gamma",
                        mu=2,
                        sigma=0.5,
                        dims=("country", "channel"),
                    ),
                    mask=mask_excluded_x4_colombia,
                ),
                "beta": Prior(
                    "Gamma",
                    mu=3,
                    sigma=0.5,
                    dims=("country", "channel"),
                ),
            }
        )

        prior = saturation.sample_prior(coords=coords, random_seed=10)
        curve = saturation.sample_curve(prior)
        saturation.plot_curve(
            curve,
            subplot_kwargs={
                "ncols": 4,
                "figsize": (12, 18),
            },
        )

    Masked likelihood over an arbitrary subset of entries (2D example over (date, country)):

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {
            "date": np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]"),
            "country": ["Venezuela", "Colombia"],
        }

        mask = xr.DataArray(
            [[True, False], [True, False]],
            dims=["date", "country"],
            coords={"date": coords["date"], "country": coords["country"]},
        )

        intercept = Prior("Normal", mu=0, sigma=10, dims=("country",))
        likelihood = Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=1), dims=("date", "country")
        )
        observed = np.random.normal(0, 1, size=(2, 2))

        with pm.Model(coords=coords):
            mu = intercept.create_variable("intercept")
            masked = MaskedPrior(likelihood, mask)
            y = masked.create_likelihood_variable("y", mu=mu, observed=observed)
    """

    def __init__(
        self, prior: Prior, mask: xr.DataArray, active_dim: str | None = None
    ) -> None:
        self.prior = prior
        self.mask = mask
        self.dims = prior.dims
        self.active_dim = active_dim or f"non_null_dims:{'_'.join(self.dims)}"
        self._validate_mask()

    def _validate_mask(self) -> None:
        if tuple(self.mask.dims) != tuple(self.dims):
            raise ValueError("mask dims must match prior.dims order")

    def _remap_dims(self, factory: VariableFactory) -> VariableFactory:
        # Depth-first remap of any nested VariableFactory with dims == parent dims
        # This keeps internal subset checks (_param_dims_work) satisfied.
        if hasattr(factory, "parameters"):
            for key, value in list(factory.parameters.items()):
                if hasattr(value, "create_variable") and hasattr(value, "dims"):
                    factory.parameters[key] = self._remap_dims(value)  # type: ignore[arg-type]

        if hasattr(factory, "dims"):
            dims = factory.dims or ()
            if isinstance(dims, str):
                dims = (dims,)
            if tuple(dims) == tuple(self.dims):
                if hasattr(factory, "parameters"):
                    flat_mask = self.mask.values.astype(bool).ravel()
                    for key, value in list(factory.parameters.items()):
                        if not (
                            hasattr(value, "create_variable") and hasattr(value, "dims")
                        ):
                            factory.parameters[key] = self._subset_raw_parameter(
                                value, flat_mask
                            )
                factory.dims = (self.active_dim,)

        return factory

    def _subset_raw_parameter(self, value: Any, flat_mask: np.ndarray) -> Any:
        """Subset a raw parameter to match the active entries of the mask.

        Dispatches on parameter type so that dimension-name-aware
        ``xr.DataArray`` values are broadcast correctly (by name),
        while plain lists / numpy arrays use positional broadcasting
        consistent with PyMC's rightmost-dim convention.
        """
        if value is None or np.isscalar(value):
            return value

        if isinstance(value, xr.DataArray):
            try:
                broadcasted, _ = xr.broadcast(value, self.mask)
                broadcasted = broadcasted.transpose(*self.mask.dims)
                return broadcasted.values.ravel()[flat_mask]
            except Exception:
                return value

        if isinstance(value, TensorVariable):
            if value.type.ndim == 0:
                return value
            try:
                broadcasted = pt.broadcast_to(value, self.mask.shape)
                return broadcasted.ravel()[flat_mask]
            except Exception:
                return value

        try:
            arr = np.asarray(value)
        except (ValueError, TypeError):
            return value
        if arr.ndim == 0:
            return value
        try:
            broadcasted = np.broadcast_to(arr, self.mask.shape)
        except ValueError:
            return value
        return broadcasted.ravel()[flat_mask]

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create a deterministic variable with full dims using the active subset.

        Creates an underlying variable over the active entries only and expands
        it back to the full masked shape, filling inactive entries with zeros.

        Parameters
        ----------
        name : str
            Base name for the created variables.

        Returns
        -------
        TensorVariable or XTensorVariable
            Deterministic variable with the original dims, zeros on inactive entries.
        """
        model = pm.modelcontext(None)
        flat_mask = self.mask.values.astype(bool).ravel()
        n_active = int(flat_mask.sum())

        if n_active == 0:
            return pm.Deterministic(name, pt.zeros(self.mask.shape), dims=self.dims)

        # Ensure the coord exists and has the right length
        if (
            self.active_dim in model.coords
            and len(model.coords[self.active_dim]) != n_active
        ):
            self.active_dim = f"{self.active_dim}__{n_active}"
        model.add_coords({self.active_dim: np.arange(n_active)})

        # Make a deep copy and remap dims depth-first before creating the RV
        reduced = self._remap_dims(self.prior.deepcopy())

        active_rv = reduced.create_variable(
            f"{name}_active", xdist=xdist
        )  # shape: (active_dim,)
        flat_zeros = pt.zeros((self.mask.size,), dtype=active_rv.dtype)
        flat_full = flat_zeros[flat_mask].set(
            as_tensor(active_rv, allow_xtensor_conversion=True)
        )
        full = flat_full.reshape(self.mask.shape)
        if xdist:
            return pmd.Deterministic(name, as_xtensor(full, dims=self.dims))
        else:
            return pm.Deterministic(name, full, dims=self.dims)

    def to_dict(self) -> dict[str, Any]:
        """Serialize MaskedPrior to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary containing the prior, mask, and active_dim.
        """
        # Store mask as a plain nested list of bools to avoid datetime coords serialization
        mask_list = (
            self.mask.values.astype(bool).tolist()
            if hasattr(self.mask, "values")
            else np.asarray(self.mask, dtype=bool).tolist()
        )
        return {
            "class": "MaskedPrior",
            "data": {
                "prior": self.prior.to_dict()
                if hasattr(self.prior, "to_dict")
                else None,
                "mask": mask_list,
                "mask_dims": list(self.dims),
                "active_dim": self.active_dim,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskedPrior":
        """Deserialize MaskedPrior from dictionary created by ``to_dict``.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        MaskedPrior
            Reconstructed instance.
        """
        data = {k: v for k, v in data.items() if k != "__type__"}
        payload = data["data"] if "data" in data else data
        prior = (
            deserialize(payload["prior"])
            if isinstance(payload.get("prior"), dict)
            else payload.get("prior")
        )
        mask_vals = payload.get("mask")
        # Fallback to provided dims or infer from prior if available
        mask_dims = payload.get("mask_dims") or (getattr(prior, "dims", None) or ())
        mask_da = xr.DataArray(np.asarray(mask_vals, dtype=bool), dims=tuple(mask_dims))
        active_dim = payload.get("active_dim")
        return cls(prior=prior, mask=mask_da, active_dim=active_dim)

    def create_likelihood_variable(
        self,
        name: str,
        *,
        mu: pt.TensorLike,
        observed: pt.TensorLike,
        xdist: bool = False,
    ) -> TensorVariable:
        """Create an observed variable over the active subset and expand to full dims.

        Parameters
        ----------
        name : str
            Base name for the created variables.
        mu : XTensorLike
            Mean/location parameter broadcastable to the masked shape.
        observed : XTensorLike
            Observations broadcastable to the masked shape.

        Returns
        -------
        XTensorVariable
            Deterministic variable over the full dims with observed RV on active entries.
        """
        if xdist:
            raise NotImplementedError(f"{self!r} only supports xdist=False")

        model = pm.modelcontext(None)
        flat_mask = self.mask.values.ravel().astype(bool)
        n_active = int(flat_mask.sum())

        if n_active == 0:
            return pm.Deterministic(name, pt.zeros(self.mask.shape), dims=self.dims)

        # Ensure the coord exists and has the right length
        if (
            self.active_dim in model.coords
            and len(model.coords[self.active_dim]) != n_active
        ):
            self.active_dim = f"{self.active_dim}__{n_active}"
        model.add_coords({self.active_dim: np.arange(n_active)})

        # Remap dims on a deep copy so nested parameter priors match the active subset
        reduced = self._remap_dims(self.prior.deepcopy())

        # Broadcast mu/observed to full mask shape via arithmetic broadcasting, then select active entries
        mu_tensor = pt.as_tensor_variable(mu)
        mu_full = mu_tensor + pt.zeros(self.mask.shape, dtype=mu_tensor.dtype)
        mu_active = mu_full.reshape((self.mask.size,))[flat_mask]

        obs = observed.values if hasattr(observed, "values") else observed
        obs_tensor = pt.as_tensor_variable(obs)
        obs_full = obs_tensor + pt.zeros(self.mask.shape, dtype=obs_tensor.dtype)
        obs_active = obs_full.reshape((self.mask.size,))[flat_mask]

        # Create the masked observed RV over the active subset
        active_name = f"{name}_active"
        active_rv = reduced.create_likelihood_variable(
            active_name, mu=mu_active, observed=obs_active
        )

        # Expand back to full shape for user-friendly access
        flat_full = pt.zeros((self.mask.size,), dtype=active_rv.dtype)
        full = flat_full[flat_mask].set(active_rv).reshape(self.mask.shape)
        return pm.Deterministic(name, full, dims=self.dims)


def _is_masked_prior_type(data: dict) -> bool:
    return data.keys() == {"class", "data"} and data.get("class") == "MaskedPrior"


register_deserialization(
    is_type=_is_masked_prior_type, deserialize=MaskedPrior.from_dict
)


def _as_prior(value: Any, field: str | None = None) -> Any:
    """Deserialize ``value`` when it is a plain dict, and return anything else as is.

    YAML ``resolve()`` and JSON loads leave priors nested inside dicts (the
    σ options, the ``mu`` of :class:`NestedZeroSumPrior`) as raw dicts. With
    ``field`` given, the result must be a prior: a ``ValueError`` naming
    ``field`` is raised otherwise.
    """
    value = deserialize(value) if isinstance(value, dict) else value
    if field and not hasattr(value, "create_variable"):
        raise ValueError(
            f"{field} must be a Prior or prior dict, got {type(value).__name__} {value!r}"
        )
    return value


@serialization.register
@dataclass
class IndependentSigmas:
    """One independent prior per level, for the scales of a :class:`NestedZeroSumPrior`.

    Parameters
    ----------
    priors : dict[str, Prior]
        The prior of each level's scale, keyed by the term and interaction
        labels of the :class:`NestedZeroSumPrior` that uses it. Each prior must
        be a scalar (no dims). Level ``<level>`` of a prior named ``<name>``
        creates ``<name>_sigma_<level>``; on the saturation ``beta``, that
        is the variable ``saturation_beta_sigma_<level>``.

    Examples
    --------
    Use it as the ``sigma`` of a :class:`NestedZeroSumPrior` on the saturation
    ``beta`` of an MMM with dims ``("brand",)``:

    .. code-block:: python

        from pymc_extras.prior import Prior

        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.mmm import MMM
        from pymc_marketing.special_priors import IndependentSigmas, NestedZeroSumPrior

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={"brand": {}},
            sigma=IndependentSigmas(priors={"brand": Prior("HalfNormal", sigma=0.2)}),
        )
        mmm = MMM(
            date_column="date",
            channel_columns=["c0", "c1"],
            target_column="y",
            dims=("brand",),
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(priors={"beta": beta}),
        )

    The same prior in a YAML config for
    :func:`~pymc_marketing.mmm.builders.yaml.build_mmm_from_yaml`:

    .. code-block:: yaml

        saturation:
          class: pymc_marketing.mmm.LogisticSaturation
          kwargs:
            priors:
              beta:
                class: pymc_marketing.special_priors.NestedZeroSumPrior
                kwargs:
                  mu: {distribution: Normal, mu: 0, sigma: 0.5, dims: channel}
                  terms:
                    brand: {}
                  sigma:
                    class: pymc_marketing.special_priors.IndependentSigmas
                    kwargs:
                      priors:
                        brand: {distribution: HalfNormal, sigma: 0.2}
    """

    priors: dict[str, Any]
    levels: list[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Deserialize the priors and record the level labels."""
        self.priors = {
            k: _as_prior(p, f"IndependentSigmas priors[{k!r}]")
            for k, p in self.priors.items()
        }
        for (
            k,
            p,
        ) in self.priors.items():  # a dimmed scale broadcasts onto the wrong axis
            if getattr(p, "dims", None):
                raise ValueError(
                    f"IndependentSigmas priors[{k!r}] must be scalar (no dims), got dims {p.dims}"
                )
        self.levels = list(self.priors)

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return {"priors": {k: p.to_dict() for k, p in self.priors.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "IndependentSigmas":
        """Deserialize from a dictionary created by :meth:`to_dict`."""
        return cls(**{k: v for k, v in data.items() if k != "__type__"})

    def create_variable(self, name: str, levels: list[str]) -> dict[str, Any]:
        """Create one scalar scale per level.

        Parameters
        ----------
        name : str
            Prefix of the variable names: level ``<level>`` is ``<name>_<level>``.
        levels : list[str]
            The level labels, in the order of the terms then the interactions.

        Returns
        -------
        dict[str, TensorVariable]
            The scale of each level, keyed by level label.
        """
        return {lv: self.priors[lv].create_variable(f"{name}_{lv}") for lv in levels}


@serialization.register
@dataclass
class VarianceBudget:
    r"""Split one total scale across the levels of a :class:`NestedZeroSumPrior`.

    .. math::

        \phi &\sim \text{Dirichlet}(\kappa \cdot \text{share}) \\
        \sigma_l &= \text{total} \cdot \sqrt{\phi_l}

    so that :math:`\sum_l \sigma_l^2 = \text{total}^2`. Adding a level
    re-divides the same total instead of adding variance, so the tails of the
    prior on :math:`\beta` do not grow with the depth of the hierarchy.

    Parameters
    ----------
    total : Prior
        Prior of the total scale, a scalar. Creates ``<name>_total``.
    share : dict[str, float]
        Center of the Dirichlet, keyed by level label. Every share must be
        greater than 0 and the shares must sum to 1.
    concentration : float, default 100
        :math:`\kappa`. Large values keep the shares near ``share``. A uniform
        ``share`` (1/L for each of L levels) with ``concentration=L`` is the flat
        Dirichlet; a small concentration around an uneven center is sparse, not
        flat.

    Notes
    -----
    ``total`` is the scale *before* the zero-sum constraint. A zero-sum vector
    of J entries with scale σ has per-entry variance :math:`\sigma^2 (1 - 1/J)`,
    so the per-cell variance of :math:`\eta` (the coefficient on the link
    scale, :math:`\log\beta` by default) around ``mu`` is
    :math:`\text{total}^2 \sum_l c_l \, \text{share}_l`, with :math:`c_l = 1 - 1/J`
    for each zero-sum axis of level :math:`l`.

    For example, on the saturation ``beta`` it creates ``saturation_beta_sigma_share``
    (the Dirichlet), ``saturation_beta_sigma_total`` and
    ``saturation_beta_sigma`` (the scales), on the dim
    ``saturation_beta_sigma_level``. The dim is named per prior so it cannot
    clash with a ``level`` coord in the model or with a second budget.

    :class:`pymc_marketing.r2d2.R2D2` uses the same Dirichlet split in a
    different direction: it splits the explained variance across the
    coefficients of one dim, while ``VarianceBudget`` splits the variance of one
    coefficient across the levels of a hierarchy.

    Examples
    --------
    Use it as the ``sigma`` of a :class:`NestedZeroSumPrior` on the saturation
    ``beta`` of an MMM with dims ``("brand",)``:

    .. code-block:: python

        from pymc_extras.prior import Prior

        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.mmm import MMM
        from pymc_marketing.special_priors import NestedZeroSumPrior, VarianceBudget

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={
                "family": {},
                "brand": {"within": "family", "map": {"b0": "f0", "b1": "f0", "b2": "f1", "b3": "f1"}},
            },
            sigma=VarianceBudget(
                total=Prior("HalfNormal", sigma=0.4),
                share={"family": 0.5, "brand": 0.5},
            ),
        )
        mmm = MMM(
            date_column="date",
            channel_columns=["c0", "c1"],
            target_column="y",
            dims=("brand",),
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(priors={"beta": beta}),
        )

    The same prior in a YAML config for
    :func:`~pymc_marketing.mmm.builders.yaml.build_mmm_from_yaml`:

    .. code-block:: yaml

        saturation:
          class: pymc_marketing.mmm.LogisticSaturation
          kwargs:
            priors:
              beta:
                class: pymc_marketing.special_priors.NestedZeroSumPrior
                kwargs:
                  mu: {distribution: Normal, mu: 0, sigma: 0.5, dims: channel}
                  terms:
                    family: {}
                    brand: {within: family, map: {b0: f0, b1: f0, b2: f1, b3: f1}}
                  sigma:
                    class: pymc_marketing.special_priors.VarianceBudget
                    kwargs:
                      total: {distribution: HalfNormal, sigma: 0.4}
                      share: {family: 0.5, brand: 0.5}
                      concentration: 100
    """

    total: Any
    share: dict[str, float]
    concentration: float = 100
    levels: list[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Deserialize ``total``, validate ``share`` and record the level labels."""
        self.total = _as_prior(self.total, "VarianceBudget total")
        if getattr(self.total, "dims", None):
            raise ValueError(
                f"VarianceBudget total must be scalar (no dims), got dims {self.total.dims}"
            )
        self.share = dict(self.share)
        if not self.share or self.concentration <= 0:
            raise ValueError(
                f"VarianceBudget needs a non-empty share and concentration > 0, "
                f"got share {self.share}, concentration {self.concentration}"
            )
        # Off the simplex, Dirichlet(concentration * share) would silently rescale the concentration
        if min(self.share.values()) <= 0 or not np.isclose(sum(self.share.values()), 1):
            raise ValueError(
                f"VarianceBudget share must be > 0 for every level and sum to 1, got {self.share}"
            )
        self.levels = list(self.share)

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return {
            "total": self.total.to_dict(),
            "share": self.share,
            "concentration": self.concentration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VarianceBudget":
        """Deserialize from a dictionary created by :meth:`to_dict`."""
        return cls(**{k: v for k, v in data.items() if k != "__type__"})

    def create_variable(self, name: str, levels: list[str]) -> dict[str, Any]:
        """Create the Dirichlet shares and the scale of each level.

        Parameters
        ----------
        name : str
            Name of the scale variable, also the prefix of ``<name>_share``,
            ``<name>_total`` and the dim ``<name>_level``.
        levels : list[str]
            The level labels, in the order of the terms then the interactions.
            The shares are matched to them by label.

        Returns
        -------
        dict[str, TensorVariable]
            The scale of each level, keyed by level label.
        """
        dim = f"{name}_level"
        pm.modelcontext(None).add_coord(dim, levels)
        concentration = self.concentration * np.array([self.share[lv] for lv in levels])
        share = pm.Dirichlet(f"{name}_share", a=concentration, dims=dim)
        total = self.total.create_variable(f"{name}_total")
        sigma = pm.Deterministic(name, total * pt.sqrt(share), dims=dim)
        return {lv: sigma[i] for i, lv in enumerate(levels)}


_TERM_KEYS = ("centered", "within", "map", "parent_effect")
_INTERACTION_KEYS = ("dims", "centered")


@serialization.register
class NestedZeroSumPrior(SpecialPrior):
    r"""Prior built from nested and crossed zero-sum offsets, for a coefficient over hierarchy dims.

    A general prior for any coefficient whose dims belong to hierarchies, for
    example brands nested in families and DMAs nested in tiers. The main
    example is the saturation ``beta`` of a multidimensional MMM, with the
    channel as base dim :math:`c`:

    .. math::

        \eta_{b,d,c} = \mu_c
        + \text{family}_{f(b),c} + \text{brand}_{b,c}
        + \text{tier}_{t(d),c} + \text{dma}_{d,c}
        + \text{brand×dma}_{b,d,c},
        \qquad \beta = \exp(\eta)

    Every offset is a ``ZeroSumNormal``: a child term is zero-sum inside each of
    its parents, a parent or crossed term over its labels, and an interaction
    over each of its dims. The zero-sum constraints remove the ridge between
    levels (a parent and the mean of its children cannot trade a shared shift).
    With ``link="log"``, ``exp`` is applied once, so the coefficient is
    positive without stacking positive distributions; with
    ``link="identity"`` it is :math:`\eta` itself, signed. Nothing is added
    implicitly: :math:`\eta` is ``mu`` plus exactly the terms and
    interactions given.

    Parameters
    ----------
    mu : Prior, float or array-like
        Prior of the mean of :math:`\eta` over the hierarchy, with the base
        dims (``("channel",)`` by default, see ``dims``). Creates
        ``<name>_mu``. A constant (a float, or one value per base label) fixes
        the mean and creates no variable.
    terms : dict[str, dict]
        The main effects, keyed by label. A spec takes the keys

        - ``centered`` (bool): parameterization of this level. Defaults to
          ``centered``.
        - ``within`` (str): the label of the parent. The term is then zero-sum
          inside each parent.
        - ``map`` (dict): child label to parent label. Required with ``within``,
          and must cover exactly the labels of the child coord. Keys are matched
          as ``str`` (a JSON round trip turns int keys into ``str``), and the
          values must be all ``str`` or all non-``str``.
        - ``parent_effect`` (bool, default True): False nests the child inside
          the parent without a parent effect. Only allowed when the parent is not
          a term.

        A term is one of three kinds:

        - A model coord with ``within``: zero-sum inside each parent. Creates one
          ``<name>_<term>_<parent label>`` per parent with at least two children;
          a lone child has offset 0.
        - A model coord without ``within``: zero-sum over the coord. Creates
          ``<name>_<term>``.
        - Not a model coord: a parent, zero-sum over its labels (the values of
          the child's ``map``). It enters through the one child that names it in
          ``within``. Creates ``<name>_<term>``.

        Nesting is at most two levels per tree: a ``within`` parent cannot
        itself be ``within`` another term, so region > tier > DMA raises.
    sigma : IndependentSigmas, VarianceBudget or Prior
        The scales of the levels, keyed by exactly the term and interaction
        labels. Terms carry no ``sigma`` of their own. Instead of the two
        options, a single-variable factory such as a pymc_extras ``Prior``
        (any object with ``dims`` and ``create_variable``) with the one dim
        ``<name>_sigma_level`` gives the scales as one vector; its entries
        are the levels, terms then interactions, in spec order. It creates
        ``<name>_sigma``, and the coord ``<name>_sigma_level`` if the model
        does not have it.
    interactions : dict[str, dict], optional
        Interactions, keyed by label: ``{"dims": [term, ...], "centered": bool}``.
        Zero-sum over every listed dim; each dim must be a term that is a model
        coord. Creates ``<name>_<label>``.
    link : {"log", "identity"}, default "log"
        The coefficient is ``exp`` of :math:`\eta` (positive) or :math:`\eta`
        itself (signed, e.g. a control coefficient or an intercept). Identity
        allows negative coefficients, so it does not suit a coefficient that
        must be positive, such as the MMM saturation ``beta``, unless intended.
    dims : tuple[str, ...], optional
        The hierarchy dims (the model-coord terms, then the interaction dims),
        then at most one base dim: a dim that is not a term, which every offset
        and ``mu`` span. Unset, the base dim is ``"channel"``, as for the MMM
        saturation ``beta``. Pass the hierarchy dims alone for no base dim
        (``mu`` scalar), or with another base dim such as ``"control"``. Any
        other dims raise.
    centered : bool, default False
        Parameterization of the levels that do not set ``centered``. Centered is
        ``ZeroSumNormal(sigma)``, non-centered ``sigma * ZeroSumNormal(1)``.

    Notes
    -----
    The spec is checked when the prior is created, and against the model coords
    in :meth:`create_variable`, before any variable exists: unknown spec keys,
    a ``sigma`` keyed by other labels, a ``map`` that misses or adds data
    labels, and colliding variable names the prior generates (the
    coefficient, the offsets, ``<name>_mu`` and the scales) all raise a
    ``ValueError`` naming the problem. A clash with a name created inside the
    ``mu`` or ``sigma`` hyperpriors surfaces as PyMC's "already exists" error.
    A level whose offsets are all 0 (every parent has one child, or a parent
    term, a crossed term or an interaction dim has one label) warns that its
    scale is unused.

    Examples
    --------
    Brands nested in families, crossed with DMAs, plus a brand × DMA
    interaction, on the saturation ``beta`` of an MMM with dims
    ``("brand", "dma")``:

    .. code-block:: python

        from pymc_extras.prior import Prior

        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.mmm import MMM
        from pymc_marketing.special_priors import IndependentSigmas, NestedZeroSumPrior

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={
                "family": {"centered": True},
                "brand": {
                    "within": "family",
                    "map": {"b0": "f0", "b1": "f0", "b2": "f1", "b3": "f1"},
                },
                "dma": {},
            },
            interactions={"brand_dma": {"dims": ["brand", "dma"]}},
            sigma=IndependentSigmas(
                priors={
                    "family": Prior("HalfNormal", sigma=0.3),
                    "brand": Prior("HalfNormal", sigma=0.3),
                    "dma": Prior("HalfNormal", sigma=0.1),
                    "brand_dma": Prior("HalfNormal", sigma=0.05),
                }
            ),
        )
        mmm = MMM(
            date_column="date",
            channel_columns=["c0", "c1"],
            target_column="y",
            dims=("brand", "dma"),
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(priors={"beta": beta}),
        )

    The same prior in a YAML config for :func:`~pymc_marketing.mmm.builders.yaml.build_mmm_from_yaml`:

    .. code-block:: yaml

        saturation:
          class: pymc_marketing.mmm.LogisticSaturation
          kwargs:
            priors:
              beta:
                class: pymc_marketing.special_priors.NestedZeroSumPrior
                kwargs:
                  mu: {distribution: Normal, mu: 0, sigma: 0.5, dims: channel}
                  terms:
                    family: {centered: true}
                    brand: {within: family, map: {b0: f0, b1: f0, b2: f1, b3: f1}}
                    dma: {}
                  interactions:
                    brand_dma: {dims: [brand, dma]}
                  sigma:
                    class: pymc_marketing.special_priors.IndependentSigmas
                    kwargs:
                      priors:
                        family: {distribution: HalfNormal, sigma: 0.3}
                        brand: {distribution: HalfNormal, sigma: 0.3}
                        dma: {distribution: HalfNormal, sigma: 0.1}
                        brand_dma: {distribution: HalfNormal, sigma: 0.05}

    The scales as one ``Prior`` vector instead, with a shared hyperprior. On
    the saturation ``beta`` the dim is ``saturation_beta_sigma_level``, here
    with the labels ``brand, dma``:

    .. code-block:: python

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={"brand": {}, "dma": {}},
            sigma=Prior(
                "HalfNormal",
                sigma=Prior("HalfNormal", sigma=0.2),
                dims="saturation_beta_sigma_level",
            ),
        )

    Outside the MMM, a signed coefficient with no base dim: an intercept per
    DMA, with DMAs nested in regions. ``mu`` is then a scalar:

    .. code-block:: python

        import pymc as pm

        intercept = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=1),
            terms={
                "region": {},
                "dma": {
                    "within": "region",
                    "map": {"d0": "r0", "d1": "r0", "d2": "r1", "d3": "r1"},
                },
            },
            sigma=IndependentSigmas(
                priors={
                    "region": Prior("HalfNormal", sigma=0.5),
                    "dma": Prior("HalfNormal", sigma=0.2),
                }
            ),
            link="identity",
            dims=("dma",),
        )
        with pm.Model(coords={"dma": ["d0", "d1", "d2", "d3"]}):
            intercept.create_variable("intercept")  # dims ("dma",)
    """

    def __init__(self, dims: tuple | None = None, centered: bool = False, **parameters):
        super().__init__(dims=dims, centered=centered, **parameters)

    def _checks(self) -> None:
        # Only checks that need no model: MMM.__init__ reads the saturation dims before any model exists.
        p = self.parameters
        if (
            not {"mu", "terms"}
            <= set(p)
            <= {"mu", "terms", "interactions", "sigma", "link"}
        ):
            raise ValueError(
                f"Parameters must be mu, terms, sigma and optionally interactions and link, got {sorted(p)}"
            )
        if p.get("link", "log") not in ("log", "identity"):
            raise ValueError(f"link must be 'log' or 'identity', got {p['link']!r}")
        if p.get("link") == "log":  # the default, left out so eq and MMM ids match it
            del p["link"]
        sigma = p.get("sigma")
        if isinstance(
            sigma, dict
        ):  # a Prior dict has no __type__, so go through pymc_extras deserialize
            sigma = _as_prior(sigma, "sigma")
        # Map keys as str, in new dicts so the caller's spec is not mutated: JSON turns int child labels into str keys
        terms = {
            c: {**t, "map": {str(k): v for k, v in t["map"].items()}}
            if "map" in t
            else dict(t)
            for c, t in dict(p["terms"]).items()
        }
        interactions = dict(p.get("interactions") or {})
        p.update(
            mu=_as_prior(p["mu"]), terms=terms, interactions=interactions, sigma=sigma
        )
        if clash := sorted(set(terms) & set(interactions)):
            raise ValueError(
                f"interactions {clash} are named like terms; term and interaction names must be unique"
            )
        specs = {**terms, **interactions}
        if own_sigma := [k for k, t in specs.items() if "sigma" in t]:
            raise ValueError(
                f"terms {own_sigma} carry their own sigma; sigma priors go in the top-level `sigma` option only"
            )
        for (
            k,
            t,
        ) in specs.items():  # a typo (`centered`, `whithin`) would otherwise be ignored
            allowed = _INTERACTION_KEYS if k in interactions else _TERM_KEYS
            if unknown := sorted(set(t) - set(allowed)):
                raise ValueError(
                    f"{k}: unknown keys {unknown}; allowed: {list(allowed)}"
                )
        if no_dims := [k for k, t in interactions.items() if "dims" not in t]:
            raise ValueError(f"interactions {no_dims} have no `dims`")
        # The prior's name, so the exact `<name>_sigma_level` dim, is only known in create_variable
        sigma_dims = tuple(getattr(sigma, "dims", None) or ())
        if (
            set(getattr(sigma, "levels", ())) != set(specs)
            if isinstance(sigma, IndependentSigmas | VarianceBudget)
            else not hasattr(sigma, "create_variable")
            or len(sigma_dims) != 1
            or not str(sigma_dims[0]).endswith("_sigma_level")
        ):
            raise ValueError(
                f"`sigma` must be IndependentSigmas or VarianceBudget keyed by exactly {sorted(specs)}, "
                f"or a prior with the single dim `<name>_sigma_level` (got dims {sigma_dims})"
            )
        parents = [t["within"] for t in terms.values() if "within" in t]
        if shared := sorted({q for q in parents if parents.count(q) > 1} & set(terms)):
            raise ValueError(
                f"terms {shared} have more than one child `within` them; a parent term enters through one child only"
            )
        for c, t in terms.items():
            if "within" not in t:
                continue
            parent = t["within"]
            if "map" not in t:
                raise ValueError(
                    f"term {c} is `within: {parent}` but has no `map` ({c} label -> {parent} label)"
                )
            if "within" in terms.get(parent, {}):
                raise ValueError(
                    f"term {c} is `within: {parent}`, which is `within: {terms[parent]['within']}`; "
                    "nesting is at most two levels per tree: a `within` parent cannot have a parent"
                )
            if parent not in terms and t.get("parent_effect", True):
                raise ValueError(
                    f"term {c} is `within: {parent}`, which is not a term: "
                    f"add {parent} to `terms`, or set `parent_effect: false` on {c}"
                )
            if parent in terms and not t.get(
                "parent_effect", True
            ):  # the parent would enter nowhere
                raise ValueError(
                    f"term {c} sets `parent_effect: false` but {parent} is a term: "
                    f"drop {parent} from `terms` and `sigma`, or drop `parent_effect`"
                )
            values = list(t["map"].values())
            if None in values:
                raise ValueError(f"term {c}: `map` has a None {parent} label")
            if (
                len({isinstance(v, str) for v in values}) > 1
            ):  # np.unique would cast 0 and "0" to one parent
                raise ValueError(
                    f"term {c}: `map` mixes str and non-str {parent} labels {sorted(set(map(repr, values)))}; "
                    "use one type"
                )
        interaction_dims = [d for t in interactions.values() for d in t["dims"]]
        if not_coords := sorted(set(interaction_dims) - (set(terms) - set(parents))):
            raise ValueError(
                f"interaction dims {not_coords} are not terms that are model coords (`within` parents are not coords)"
            )
        hierarchy = tuple(
            dict.fromkeys([k for k in terms if k not in parents] + interaction_dims)
        )
        if self.dims is None and "channel" in terms:
            raise ValueError(
                "term 'channel' is named like the default base dim 'channel'; pass `dims` explicitly"
            )
        dims = (
            (*hierarchy, "channel")
            if self.dims is None
            else (self.dims,)
            if isinstance(self.dims, str)
            else tuple(self.dims)
        )
        if len(base := [d for d in dims if d not in terms]) > 1:
            raise ValueError(
                f"dims {dims} have more than one base dim {base}; at most one dim may be outside the terms"
            )
        if dims != (*hierarchy, *base):
            raise ValueError(
                f"dims {dims} must be {hierarchy}, the dims the terms and interactions imply, "
                f"then at most one base dim; or leave dims unset for {(*hierarchy, 'channel')}"
            )
        mu_dims = tuple(getattr(p["mu"], "dims", None) or ())
        if hasattr(p["mu"], "create_variable") and mu_dims != tuple(base):
            raise ValueError(
                f"mu has dims {mu_dims}; it must have the base dims {tuple(base)}"
            )
        if not hasattr(
            p["mu"], "create_variable"
        ):  # a constant: one number, or one per base label
            mu = np.asarray(p["mu"])
            if mu.dtype.kind not in "iuf" or mu.ndim > len(base):
                raise ValueError(
                    f"constant mu must be a number, or a list with one number per label "
                    f"of the base dims {tuple(base)}, got {p['mu']!r}"
                )
            p["mu"] = (
                mu.tolist()
            )  # an ndarray as a list, so from_dict(to_dict()) == self
        self.dims = dims

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create the coefficient from the zero-sum offsets, after checking the spec against the model coords.

        Parameters
        ----------
        name : str
            Name of the coefficient's Deterministic, and prefix of every variable it
            creates.
        xdist : bool, default False
            Return the XTensorVariable instead of its tensor.

        Returns
        -------
        TensorVariable or XTensorVariable
            The coefficient, with dims :attr:`dims`.
        """
        mu, terms, interactions, sigma = (
            self.parameters[k] for k in ("mu", "terms", "interactions", "sigma")
        )
        coords = pm.modelcontext(None).coords
        specs = {**terms, **interactions}
        base = tuple(d for d in self.dims if d not in terms)  # type: ignore[union-attr]  # set by _checks
        if missing := [d for d in base if d not in coords]:
            raise ValueError(
                f"the model has no `{missing[0]}` coord; coords: {sorted(coords)}"
            )
        n_base = tuple(len(coords[d]) for d in base)
        levels, level_dim = list(specs), f"{name}_sigma_level"
        is_option = isinstance(sigma, IndependentSigmas | VarianceBudget)
        if not is_option and tuple(sigma.dims) != (level_dim,):
            raise ValueError(
                f"`sigma` has dims {tuple(sigma.dims)}; for {name} it must have the single dim {level_dim!r}"
            )
        if not is_option and level_dim in coords and list(coords[level_dim]) != levels:
            raise ValueError(
                f"model coord {level_dim!r} has labels {list(coords[level_dim])}, "
                f"but the levels of {name} (terms, then interactions) are {levels}"
            )
        # Data checks before any variable is created
        for c, t in terms.items():
            if c not in coords and not any(
                u.get("within") == c for u in terms.values()
            ):
                raise ValueError(
                    f"term {c}: not a model coord {sorted(coords)} and no term is `within` it"
                )
            if "within" not in t:
                continue
            parent = t["within"]
            if c not in coords:
                raise ValueError(
                    f"term {c} is `within: {parent}`, so it must be a model coord; coords: {sorted(coords)}"
                )
            if parent in coords:
                raise ValueError(
                    f"term {c} is `within: {parent}`, which is a model coord; "
                    f"a `within` parent groups {c} labels, not a coord"
                )
            if missing := [lab for lab in coords[c] if str(lab) not in t["map"]]:
                raise ValueError(
                    f"term {c}: `map` has no {parent} for data labels {missing}"
                )
            if extra := sorted(set(t["map"]) - {str(lab) for lab in coords[c]}):
                raise ValueError(
                    f"term {c}: `map` keys {extra} are not {c} labels in the data {list(coords[c])}"
                )
        # Per nested term: the parent labels, and the parent index of each child in the coord order
        groups = {
            c: np.unique([t["map"][str(lab)] for lab in coords[c]], return_inverse=True)
            for c, t in terms.items()
            if "within" in t
        }
        if not hasattr(mu, "create_variable") and np.shape(mu) not in ((), n_base):
            raise ValueError(
                f"constant mu has shape {np.shape(mu)}; expected () or {n_base}, one value per {base} label"
            )
        # The names this prior generates, checked before any exists; hyperprior names are left to pymc
        names = [
            name,
            *(f"{name}_{k}" for k, t in specs.items() if "within" not in t),
            *(
                [f"{name}_sigma_{lv}" for lv in levels]
                if isinstance(sigma, IndependentSigmas)
                else [f"{name}_sigma"]
            ),
            *([f"{name}_mu"] if hasattr(mu, "create_variable") else []),
            *(
                f"{name}_{c}_{lab}"
                for c, (labels, codes) in groups.items()
                for lab in labels[np.bincount(codes) > 1]
            ),
        ]
        if duplicates := sorted({r for r in names if names.count(r) > 1}):
            raise ValueError(
                f"RV names {duplicates} are generated twice: rename a term, interaction or parent label"
            )
        for c, (_, codes) in groups.items():  # a scale that only multiplies zeros
            parent = terms[c]["within"]
            if np.bincount(codes).max() < 2:
                warnings.warn(
                    f"term {c}: every {parent} has one {c}, so all {c} offsets are 0 and its sigma is unused",
                    stacklevel=2,
                )
        one_label = [  # parent terms, then crossed terms
            *(
                t["within"]
                for c, t in terms.items()
                if c in groups and t["within"] in terms and len(groups[c][0]) == 1
            ),
            *(
                c
                for c, t in terms.items()
                if "within" not in t and c in coords and len(coords[c]) == 1
            ),
        ]
        for c in one_label:
            warnings.warn(
                f"term {c} has one label, so its effect is 0 and its sigma is unused",
                stacklevel=2,
            )
        for k, t in interactions.items():
            if one := [d for d in t["dims"] if len(coords[d]) == 1]:
                warnings.warn(
                    f"interaction {k}: dims {one} have one label, so its offsets are 0 and its sigma is unused",
                    stacklevel=2,
                )
        if is_option:
            scales = sigma.create_variable(f"{name}_sigma", levels)
        else:
            # A no-op when the labels match, e.g. when a lazily built R2D2 decomposition added the coord first
            pm.modelcontext(None).add_coord(level_dim, levels)
            sigma_var = sigma.create_variable(f"{name}_sigma")
            sigma_var = getattr(
                sigma_var, "values", sigma_var
            )  # XTensorVariable -> TensorVariable
            # A factory may ignore its dims, so check the length it actually returned
            if (shape := tuple(sigma_var.shape.eval().tolist())) != (len(levels),):
                raise ValueError(
                    f"`sigma` returned shape {shape}; "
                    f"expected ({len(levels)},), one scale per level {levels}"
                )
            scales = {lv: sigma_var[i] for i, lv in enumerate(levels)}

        def zero_sum(level, rv_name, shape, n_zerosum_axes=1):
            centered = specs[level].get("centered", self.centered)
            offset = pm.ZeroSumNormal(
                f"{name}_{rv_name}",
                sigma=scales[level] if centered else 1.0,
                shape=shape,
                n_zerosum_axes=n_zerosum_axes,
            )
            return offset if centered else scales[level] * offset

        mu = (  # a constant mu: a float or one value per base label
            mu.create_variable(f"{name}_mu")
            if hasattr(mu, "create_variable")
            else pt.broadcast_to(pt.as_tensor(mu), n_base)
        )
        eta = as_xtensor(mu, dims=base)  # beta on the link scale
        for c, t in terms.items():
            if "within" in t:
                labels, codes = groups[c]
                offset = pt.zeros((len(codes), *n_base))
                for p, label in enumerate(
                    labels
                ):  # one zero-sum vector per parent; lone children stay 0
                    idx = np.flatnonzero(codes == p)
                    if len(idx) > 1:
                        offset = pt.set_subtensor(
                            offset[idx],
                            zero_sum(c, f"{c}_{label}", (*n_base, len(idx))).T,
                        )
                if t["within"] in terms:  # otherwise parent_effect is False (_checks)
                    offset = (
                        offset
                        + zero_sum(t["within"], t["within"], (*n_base, len(labels))).T[
                            codes
                        ]
                    )
                eta = eta + as_xtensor(offset, dims=(c, *base))
            elif c in coords:
                eta = eta + as_xtensor(
                    zero_sum(c, c, (*n_base, len(coords[c]))), dims=(*base, c)
                )
        for k, t in interactions.items():
            shape = (*n_base, *(len(coords[d]) for d in t["dims"]))
            eta = eta + as_xtensor(
                zero_sum(k, k, shape, len(t["dims"])), dims=(*base, *t["dims"])
            )
        if self.parameters.get("link", "log") == "log":
            eta = pmd.math.exp(eta)
        beta = pmd.Deterministic(name, eta.transpose(*self.dims))  # type: ignore[misc]
        return beta if xdist else beta.values

    def to_dict(self):
        """Convert the prior to a dictionary."""
        # SpecialPrior.to_dict writes `centered` only when it is False and `dims` only when not empty,
        # but here the default centered is False and empty dims are not the default
        return super().to_dict() | {"centered": self.centered, "dims": self.dims}

    def __hash__(self):
        """Hash the parts __eq__ compares that are order-free (SpecialPrior hashes str(dict), so key order)."""
        return hash(
            (
                type(self).__name__,
                self.dims,
                self.centered,
                frozenset(self.parameters["terms"]),
                frozenset(self.parameters["interactions"]),
            )
        )

    @classmethod
    def from_dict(cls, data) -> "NestedZeroSumPrior":
        """Create the prior from a dictionary created by :meth:`to_dict`."""
        # SpecialPrior.from_dict would deserialize() every dict argument, `terms` and `interactions` included
        return cls(
            dims=data.get("dims"),
            centered=data.get("centered", False),
            **data["kwargs"],
        )


# MMM.load reads the saturation priors back through pymc_extras.deserialize, which does not consult
# pymc-marketing's registry. NestedZeroSumPrior is found by `_deserialize_special_prior`,
# the sigma options by `__type__`.
for _cls in (IndependentSigmas, VarianceBudget):
    register_deserialization(
        is_type=lambda d, _c=_cls: (
            isinstance(d, dict)
            and d.get("__type__") == f"{_c.__module__}.{_c.__qualname__}"
        ),
        deserialize=_cls.from_dict,
    )


def _is_special_prior_type(data: dict) -> bool:
    """Check if data represents a SpecialPrior subclass."""
    return isinstance(data, dict) and "special_prior" in data


def _get_all_special_prior_subclasses(
    base_class: type[SpecialPrior],
) -> dict[str, type[SpecialPrior]]:
    """Recursively get all subclasses of a base class.

    Returns a dict mapping class name to class object.
    """
    subclasses: dict[str, type[SpecialPrior]] = {}
    for subclass in base_class.__subclasses__():
        subclasses[subclass.__name__] = subclass
        # Recursively get subclasses of subclasses
        subclasses.update(_get_all_special_prior_subclasses(subclass))
    return subclasses


def _deserialize_special_prior(data: dict) -> SpecialPrior:
    """Deserialize any SpecialPrior subclass by looking up the class dynamically.

    This function automatically discovers all SpecialPrior subclasses using __subclasses__(),
    so new SpecialPrior subclasses don't need explicit registration.
    """
    class_name = data.get("special_prior")
    if not isinstance(class_name, str):
        raise ValueError(
            f"Expected 'special_prior' to be a string, got {type(class_name)}"
        )

    # Get all SpecialPrior subclasses recursively
    special_prior_classes = _get_all_special_prior_subclasses(SpecialPrior)  # type: ignore[type-abstract]

    cls = special_prior_classes.get(class_name)
    if cls is None:
        raise ValueError(
            f"Unknown SpecialPrior class: {class_name}. "
            f"Available classes: {list(special_prior_classes.keys())}"
        )

    return cls.from_dict(data)


register_deserialization(
    is_type=_is_special_prior_type, deserialize=_deserialize_special_prior
)


def is_alternative_prior(data: Any) -> bool:
    """Check if the data is a dictionary representing a Prior (alternative check)."""
    return isinstance(data, dict) and isinstance(data.get("distribution"), str)


def deserialize_alternative_prior(data: dict[str, Any]) -> Prior:
    """Alternative deserializer that recursively handles all nested parameters.

    This handles the flat ``{"distribution": ...}`` prior format used by the MMM
    YAML schema, where any parameter might itself be a nested prior, and also
    extracts the ``centered`` and ``transform`` parameters.

    Examples
    --------
    This handles cases like:

    .. code-block:: yaml

        distribution: Gamma
        alpha: 1
        beta:
            distribution: HalfNormal
            sigma: 1
            dims: channel
        dims: [brand, channel]

    """
    data = copy.deepcopy(data)

    distribution = data.pop("distribution")
    dims = data.pop("dims", None)
    centered = data.pop("centered", True)
    transform = data.pop("transform", None)
    parameters = data

    # Recursively deserialize any nested parameters
    parameters = {
        key: value if not isinstance(value, dict) else deserialize(value)
        for key, value in parameters.items()
    }

    return Prior(
        distribution,
        transform=transform,
        centered=centered,
        dims=dims,
        **parameters,
    )


register_deserialization(is_alternative_prior, deserialize_alternative_prior)
