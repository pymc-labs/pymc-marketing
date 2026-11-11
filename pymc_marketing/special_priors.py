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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Self

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
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
    _get_transform,
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


def _level_dim(dims: str | tuple[str, ...] | None, name: str) -> str:
    """Return the one level dim of :class:`IndependentSigmas` or :class:`VarianceBudget`, or raise."""
    if dims is None or len(dims) != 1:
        raise ValueError(
            f"{name}: `dims` must be one level dim, got {dims}; "
            "a NestedZeroSumPrior sets it to `<name>_sigma_level`"
        )
    return dims[0]


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
        creates ``<name>_sigma_<level>``, and the levels are stacked into the
        vector ``<name>_sigma`` on the dim ``<name>_sigma_level``; on the
        saturation ``beta``, those are ``saturation_beta_sigma_<level>`` and
        ``saturation_beta_sigma``.
    dims : str or tuple[str], optional
        The level dim, whose model coord gives the levels and their order. Leave
        it unset in a :class:`NestedZeroSumPrior`, which sets it on a copy.

    Notes
    -----
    A pymc_extras ``VariableFactory``: :meth:`create_variable` returns one
    vector over the level dim, like a ``Prior`` with that dim, so it can also
    be a parameter of a ``Prior``.

    Examples
    --------
    Use it as the ``sigma`` of a :class:`NestedZeroSumPrior` on the saturation
    ``beta`` of an MMM with dims ``("brand",)``:

    .. code-block:: python

        from pymc_extras.prior import Prior

        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.mmm import MMM
        from pymc_marketing.special_priors import (
            IndependentSigmas,
            NestedZeroSumPrior,
            Term,
        )

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={"brand": Term()},
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
    dims: str | tuple[str, ...] | None = None
    levels: list[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Deserialize the priors, normalize ``dims`` and record the level labels."""
        self.dims = None if self.dims is None else _as_dims(self.dims)
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
        return {
            "priors": {k: p.to_dict() for k, p in self.priors.items()},
            **({} if self.dims is None else {"dims": list(self.dims)}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IndependentSigmas":
        """Deserialize from a dictionary created by :meth:`to_dict`."""
        return cls(**{k: v for k, v in data.items() if k != "__type__"})

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create one scalar scale per level, stacked into one vector.

        Parameters
        ----------
        name : str
            Name of the vector, on :attr:`dims`, and prefix of the scale of
            each level, ``<name>_<level>``. The levels are matched to
            ``priors`` by label.
        xdist : bool, default False
            Return the XTensorVariable instead of its tensor.

        Returns
        -------
        TensorVariable or XTensorVariable
            The scales, with dims :attr:`dims`.
        """
        dim = _level_dim(self.dims, name)
        scales = [
            self.priors[lv].create_variable(f"{name}_{lv}", xdist=True).expand_dims(dim)
            for lv in pm.modelcontext(None).coords[dim]
        ]
        sigma = pmd.Deterministic(name, pmd.concat(scales, dim=dim))
        return sigma if xdist else sigma.values


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
    dims : str or tuple[str], optional
        The level dim, as for :class:`IndependentSigmas`. Leave it unset in a
        :class:`NestedZeroSumPrior`, which sets it on a copy.

    Notes
    -----
    ``total`` is the scale *before* the zero-sum constraint. A zero-sum vector
    of J entries with scale σ has per-entry variance :math:`\sigma^2 (1 - 1/J)`,
    so the per-cell variance of :math:`\eta` (the coefficient before the
    ``transform``, :math:`\log\beta` by default) around ``mu`` is
    :math:`\text{total}^2 \sum_l c_l \, \text{share}_l`, with :math:`c_l = 1 - 1/J`
    for each zero-sum axis of level :math:`l`.

    For example, on the saturation ``beta`` it creates ``saturation_beta_sigma_share``
    (the Dirichlet), ``saturation_beta_sigma_total`` and
    ``saturation_beta_sigma`` (the scales), on the dim
    ``saturation_beta_sigma_level``. The dim is named per prior so it cannot
    clash with a ``level`` coord in the model or with a second budget. Like
    :class:`IndependentSigmas`, it is a pymc_extras ``VariableFactory`` over
    the level dim ``dims``.

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
        from pymc_marketing.special_priors import NestedZeroSumPrior, Term, VarianceBudget

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={
                "family": Term(),
                "brand": Term(within="family", map={"b0": "f0", "b1": "f0", "b2": "f1", "b3": "f1"}),
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
    dims: str | tuple[str, ...] | None = None
    levels: list[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Deserialize ``total``, validate ``share``, normalize ``dims`` and record the level labels."""
        self.dims = None if self.dims is None else _as_dims(self.dims)
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
            **({} if self.dims is None else {"dims": list(self.dims)}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VarianceBudget":
        """Deserialize from a dictionary created by :meth:`to_dict`."""
        return cls(**{k: v for k, v in data.items() if k != "__type__"})

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create the Dirichlet shares and the scale of each level.

        Parameters
        ----------
        name : str
            Name of the scale variable, on :attr:`dims`, and prefix of
            ``<name>_share`` and ``<name>_total``. The shares are matched to the
            levels by label.
        xdist : bool, default False
            Return the XTensorVariable instead of its tensor.

        Returns
        -------
        TensorVariable or XTensorVariable
            The scales, with dims :attr:`dims`.
        """
        dim = _level_dim(self.dims, name)
        share = [self.share[lv] for lv in pm.modelcontext(None).coords[dim]]
        share = pmd.Dirichlet(
            f"{name}_share",
            a=as_xtensor(self.concentration * np.array(share), dims=(dim,)),
            core_dims=(dim,),
            dims=(dim,),
        )
        total = self.total.create_variable(f"{name}_total", xdist=True)
        sigma = pmd.Deterministic(name, total * pmd.math.sqrt(share))
        return sigma if xdist else sigma.values


class _GroupedZeroSumOffset:
    r"""An offset over ``child_dim`` that sums to zero within each group of children, one ``Normal`` in all.

    A pymc_extras ``VariableFactory`` built by :class:`NestedZeroSumPrior`.
    A ``Normal`` vector less the mean of each group has the distribution of one ``ZeroSumNormal`` per group.
    The number of variables does not depend on the number of groups, and no group label becomes a name.
    The group means are a product with the one-hot matrix of the groups, of size children times groups.
    A group with one child is 0.

    Parameters
    ----------
    scale : float or XTensorVariable
        The scale of the ``Normal`` before the group mean is removed.
        An entry of a group of :math:`n` children then has the standard deviation :math:`scale \sqrt{1 - 1/n}`.
    centered : bool
        The parameterization of the ``Normal``, as in ``Prior(..., centered=...)``.
    dims : tuple[str, ...]
        The dims of the offset, including ``child_dim``.
    child_dim : str
        The dim of the children.
    codes : array-like of int
        The group of each child, from 0, in the order of the labels of ``child_dim``.
    """

    def __init__(
        self,
        scale: float | XTensorVariable,
        centered: bool,
        dims: tuple[str, ...],
        child_dim: str,
        codes: ArrayLike,
    ) -> None:
        self.scale, self.centered = scale, centered
        self.dims, self.child_dim = tuple(dims), child_dim
        self.codes = np.asarray(codes)

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create the offset as ``name``, from the ``Normal`` ``<name>_raw``."""
        group_dim = f"_{self.child_dim}_group"  # only in the graph, not a model coord
        onehot = as_xtensor(
            np.eye(self.codes.max() + 1, dtype=pytensor.config.floatX)[self.codes],
            dims=(self.child_dim, group_dim),
        )
        raw = Prior(
            "Normal", mu=0, sigma=self.scale, centered=self.centered, dims=self.dims
        ).create_variable(f"{name}_raw", xdist=True)
        group_mean = pmd.dot(raw, onehot, dim=self.child_dim) / onehot.sum(
            self.child_dim
        )
        # each child's group mean, by index: no second product with the one-hot matrix
        spread = group_mean.isel({group_dim: self.codes}).rename(
            {group_dim: self.child_dim}
        )
        offset = pmd.Deterministic(name, (raw - spread).transpose(*self.dims))
        return offset if xdist else offset.values


def _non_scalar_params(prior: Prior, path: str) -> list[str]:
    """Return the parameters of ``prior``, and of the priors inside it, that hold more than one value."""
    found = []
    for key, value in prior.parameters.items():
        if isinstance(value, Prior):
            found += _non_scalar_params(value, f"{path}.{key}")
        elif not hasattr(value, "create_variable") and np.ndim(value) > 0:
            found.append(f"{path}.{key}")
    return found


def _as_dims(dims: Any) -> tuple:
    """Dims as a tuple: a str is one dim (not its characters), None is no dims."""
    return () if dims is None else (dims,) if isinstance(dims, str) else tuple(dims)


@dataclass
class _Spec:
    """Base of :class:`Term` and :class:`Interaction`: built from a plain dict, written back as one."""

    @classmethod
    def coerce(cls, label: str, spec: Any) -> Self:
        """Return ``spec`` as a ``cls``, converting a plain dict and naming ``label`` in its errors."""
        if isinstance(spec, cls):
            return spec
        spec = {} if spec is None else spec  # YAML `brand:` with no value
        allowed = [f.name for f in fields(cls)]
        # e.g. YAML `brand: family`, whose characters would otherwise be read as keys
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"{cls.__name__.lower()} {label}: expected a mapping with the keys {allowed}, "
                f"got the {type(spec).__name__} {spec!r}"
            )
        if unknown := sorted(
            set(spec) - set(allowed)
        ):  # a typo would otherwise be ignored
            raise ValueError(
                f"{cls.__name__.lower()} {label}: unknown keys {unknown}; allowed: {allowed}"
            )
        try:
            return cls(**spec)
        except ValueError as err:
            raise ValueError(f"{cls.__name__.lower()} {label}: {err}") from None

    def to_dict(self) -> dict:
        """Serialize to a plain dict, leaving out the fields at their default."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) != f.default
        }


@dataclass
class Term(_Spec):
    """A main effect of a :class:`NestedZeroSumPrior`, the value of one entry of its ``terms``.

    A plain dict with the same keys is converted to a ``Term``, so YAML and
    JSON configs stay plain dicts. A term with no value (YAML ``brand:``) is
    ``Term()``.

    Parameters
    ----------
    centered : bool, optional
        Parameterization of this level. Unset, the ``centered`` of the prior.
    within : str, optional
        The label of the parent. The term is then zero-sum inside each parent.
    map : dict, optional
        Child label to parent label. Required with ``within``, and must cover
        exactly the labels of the child coord. Keys are matched as ``str`` (a
        JSON round trip turns int keys into ``str``), and the values must be
        all ``str`` or all non-``str``.
    parent_effect : bool, default True
        False nests the child inside the parent without a parent effect. Only
        allowed when the parent is not a term.
    """

    centered: bool | None = None
    within: str | None = None
    map: dict | None = None
    parent_effect: bool = True

    def __post_init__(self) -> None:
        """Validate the term on its own; the checks across terms are in :class:`NestedZeroSumPrior`."""
        if not (
            isinstance(self.centered, bool | None)
            and isinstance(self.within, str | None)
            and isinstance(self.parent_effect, bool)
        ):
            raise ValueError(
                "`centered` must be a bool or None, `within` a str or None and `parent_effect` a bool; got "
                f"centered={self.centered!r}, within={self.within!r}, parent_effect={self.parent_effect!r}"
            )
        if self.within is None and (self.map is not None or not self.parent_effect):
            raise ValueError("`map` and `parent_effect` need `within`")
        if self.within is not None and self.map is None:
            raise ValueError(
                f"is `within: {self.within}` but has no `map` (child label -> {self.within} label)"
            )
        # In a new dict, so the caller's map is not mutated: JSON turns int child labels into str keys
        self.map = (
            None if self.map is None else {str(k): v for k, v in self.map.items()}
        )
        values = list((self.map or {}).values())
        if None in values:
            raise ValueError(f"`map` has a None {self.within} label")
        if (
            len({isinstance(v, str) for v in values}) > 1
        ):  # np.unique would cast 0 and "0" to one parent
            raise ValueError(
                f"`map` mixes str and non-str {self.within} labels {sorted(set(map(repr, values)))}; use one type"
            )


@dataclass
class Interaction(_Spec):
    """An interaction of a :class:`NestedZeroSumPrior`, the value of one entry of its ``interactions``.

    A plain dict with the same keys is converted to an ``Interaction``.
    An interaction with no value (YAML ``brand_dma:``) has no ``dims`` and
    raises.

    Parameters
    ----------
    dims : list[str]
        The terms it interacts, each a term that is a model coord. The offset
        is zero-sum over every one of them.
    centered : bool, optional
        Parameterization of this level. Unset, the ``centered`` of the prior.
    """

    dims: list[str] = field(default_factory=list)
    centered: bool | None = None

    def __post_init__(self) -> None:
        """Validate the interaction on its own."""
        self.dims = [self.dims] if isinstance(self.dims, str) else list(self.dims or [])
        if not self.dims or len(set(self.dims)) < len(self.dims):
            raise ValueError(
                f"`dims` must be non-empty, each dim once; got {self.dims}"
            )
        if not isinstance(self.centered, bool | None):
            raise ValueError(
                f"`centered` must be a bool or None, got {self.centered!r}"
            )


@serialization.register
class NestedZeroSumPrior:
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

    Every offset is zero-sum.
    A child term sums to zero inside each of its parents, and a parent or crossed term over its labels.
    An interaction sums to zero over each of its dims.
    A parent, crossed term or interaction is a ``ZeroSumNormal``.
    A child term is one ``Normal`` over all the children, less the mean of each parent's children.
    This has the distribution of one ``ZeroSumNormal`` per parent, without one variable per parent.
    The zero-sum constraints remove the ridge between
    levels (a parent and the mean of its children cannot trade a shared shift).
    With the default ``transform="exp"``, ``exp`` is applied once, so the
    coefficient is positive without stacking positive distributions; with
    ``transform=None`` it is :math:`\eta` itself, signed. Nothing is added
    implicitly: :math:`\eta` is ``mu`` plus exactly the terms and
    interactions given.

    A pymc_extras ``VariableFactory`` (``dims`` and :meth:`create_variable`).

    Parameters
    ----------
    mu : Prior, float or array-like
        Prior of the mean of :math:`\eta` over the hierarchy, with the base
        dim (see ``dims``) or none. Creates ``<name>_mu``. A constant (a float,
        or one value per base label) fixes the mean and creates no variable.
    terms : dict[str, Term or dict]
        The main effects, keyed by label. Each is a :class:`Term`; a plain dict
        with the same keys (``centered``, ``within``, ``map``,
        ``parent_effect``) is also accepted, as YAML and JSON configs give it.
        A term is one of three kinds:

        - A model coord with ``within``: zero-sum inside each parent.
          Creates the ``Normal`` ``<name>_<term>_raw`` and the offset ``<name>_<term>``, on the coord ``<term>``.
          The number of variables does not depend on the number of parents, and no parent label is part of a name.
          A lone child has offset 0.
        - A model coord without ``within``: zero-sum over the coord. Creates
          ``<name>_<term>``.
        - Not a model coord: a parent, zero-sum over its labels (the values of
          the child's ``map``). It enters through the one child that names it in
          ``within``. Creates ``<name>_<term>``, on the coord ``<term>`` of its
          labels (added to the model; an existing coord must have those labels).

        Nesting is at most two levels per tree: a ``within`` parent cannot
        itself be ``within`` another term, so region > tier > DMA raises.
        The labels of terms and interactions are part of variable names, so they cannot contain ``/``.
        The parent labels, the values of a ``map``, can.
    sigma : IndependentSigmas, VarianceBudget or Prior
        The scales of the levels, the term and interaction labels. A
        single-variable factory: called as ``<name>_sigma``, it returns one
        vector on the dim ``<name>_sigma_level``, whose labels are the levels,
        the term and interaction labels, sorted. :class:`IndependentSigmas`
        (one prior per level) and :class:`VarianceBudget` (one total split
        across the levels) are keyed by exactly the level labels and leave
        ``dims`` unset. Any other factory, such as a pymc_extras ``Prior``,
        is given without dims (hyperpriors allowed): a copy of it gets the dim
        ``<name>_sigma_level`` when the variable is created. The coord
        ``<name>_sigma_level`` is added if the model does not have it.
        The scales must be positive, and, as in PyMC, this is not checked.
        A prior that can be negative, such as a ``Normal``, gives a ``nan`` log density when centered.
        When not centered, it gives two mirror modes, since a scale and its offset can both change sign.
        Use a prior with positive support, such as ``HalfNormal`` or ``Gamma``.
    interactions : dict[str, Interaction or dict], optional
        Interactions, keyed by label. Each is an :class:`Interaction`; a plain
        dict with the same keys, ``{"dims": [term, ...], "centered": bool}``,
        is also accepted, as YAML and JSON configs give it. Zero-sum over every listed
        dim; each dim must be a term that is a model coord. Creates
        ``<name>_<label>``.
    transform : str or None, default "exp"
        Applied to :math:`\eta`: any transform name a pymc_extras ``Prior``
        accepts (a ``pytensor.tensor`` or ``pymc.math`` function, or one
        registered with ``pymc_extras.prior.register_tensor_transform``), or
        None for :math:`\eta` itself (signed, e.g. a control coefficient or an
        intercept). Unlike ``Prior``, whose default is None, the default is
        ``"exp"``, so the coefficient is positive, as the MMM saturation
        ``beta`` must be.
    centered : bool, default False
        Parameterization of the levels that do not set ``centered``. Centered is
        ``ZeroSumNormal(sigma)``, non-centered ``sigma * ZeroSumNormal(1)``.
        A child term (``within``) is drawn the same way as a ``Normal``, before each parent's mean is removed.
        Prefer the default for child terms: see the Notes.
    dims : tuple[str, ...], optional
        The dims of the coefficient, in any order: the hierarchy dims (the
        model-coord terms and the interaction dims) and at most one base dim,
        a dim that is not a term, which every offset and ``mu`` span. Inside
        an MMM, leave it unset: the MMM sets it to ``(*mmm.dims, "channel")``,
        and its dims must then be the hierarchy dims. Pass ``dims`` explicitly
        for a hierarchy over only some of the MMM dims (the coefficient then
        broadcasts over the others), and outside the MMM.

    Notes
    -----
    The spec is checked when the prior is created, ``dims`` when they are set,
    and the rest against the model coords in :meth:`create_variable`, before
    any variable exists: unknown spec keys, a ``sigma`` keyed by other labels,
    a ``map`` that misses or adds data labels, colliding variable and coord
    names the prior generates (the coefficient, the offsets, ``<name>_mu``,
    ``<name>_sigma`` and the coords it adds), and a model coord it adds that
    already exists with other labels all raise a ``ValueError`` naming the
    problem. A clash
    with a name created inside the ``mu`` or ``sigma`` factories (such as the
    ``<name>_sigma_<level>`` of :class:`IndependentSigmas`) surfaces as PyMC's
    "already exists" error. A level whose offsets are all 0 (every parent has
    one child, or a parent term, a crossed term or an interaction dim has one
    label) warns that its scale is unused.

    A child term (``within``) is one ``Normal`` over all of its children, less the mean of each parent's children.
    It is not one ``ZeroSumNormal`` per parent, but it has the same distribution.
    The group of :math:`n` children has the covariance :math:`\sigma^2 (I - \mathbf{1}\mathbf{1}^\top / n)`.
    That is the covariance of a ``ZeroSumNormal``, and the groups are independent.
    Only the variables differ, not the distribution of the offsets.
    The number of variables does not depend on the number of parents.
    No parent label is part of a variable or coord name.
    To select the offsets of one parent from the draws, use :meth:`children`.
    The group means are a product with the one-hot matrix of the groups, of size children times parents.
    That is small for hundreds of labels, but it grows with both.

    The mean direction of each parent is not informed by the data.
    With ``centered=True`` its prior scale is the ``sigma`` of the level, which couples it to the scale being learned.
    In a simulation study (informative data, four seeds, NUTS), centered child terms sampled the level scales worse.
    Their effective sample size (``<name>_sigma``) was about half that of one ``ZeroSumNormal`` per parent.
    The coefficient itself sampled as well or better, and there were no divergences.
    Non-centered child terms, the default, sampled the scales as well as the per-parent form.
    Prefer ``centered=False`` for child terms, as usual for hierarchies with little data per label.
    The study used one model shape and informative data, and sparse data was not tested.

    Examples
    --------
    Brands nested in families, crossed with DMAs, plus a brand × DMA
    interaction, on the saturation ``beta`` of an MMM with dims
    ``("brand", "dma")``:

    .. code-block:: python

        from pymc_extras.prior import Prior

        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.mmm import MMM
        from pymc_marketing.special_priors import (
            IndependentSigmas,
            Interaction,
            NestedZeroSumPrior,
            Term,
        )

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={
                "family": Term(centered=True),
                "brand": Term(
                    within="family",
                    map={"b0": "f0", "b1": "f0", "b2": "f1", "b3": "f1"},
                ),
                "dma": Term(),
            },
            interactions={"brand_dma": Interaction(dims=["brand", "dma"])},
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

    The scales as one ``Prior`` vector instead, with a shared hyperprior. It is
    given without dims; on the saturation ``beta`` it gets the dim
    ``saturation_beta_sigma_level``, here with the labels ``brand, dma``:

    .. code-block:: python

        beta = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
            terms={"brand": Term(), "dma": Term()},
            sigma=Prior("HalfNormal", sigma=Prior("HalfNormal", sigma=0.2)),
        )

    Outside the MMM, a signed coefficient with no base dim: an intercept per
    DMA, with DMAs nested in regions. ``mu`` is then a scalar:

    .. code-block:: python

        import pymc as pm

        intercept = NestedZeroSumPrior(
            mu=Prior("Normal", mu=0, sigma=1),
            terms={
                "region": Term(),
                "dma": Term(
                    within="region",
                    map={"d0": "r0", "d1": "r0", "d2": "r1", "d3": "r1"},
                ),
            },
            sigma=IndependentSigmas(
                priors={
                    "region": Prior("HalfNormal", sigma=0.5),
                    "dma": Prior("HalfNormal", sigma=0.2),
                }
            ),
            transform=None,
            dims=("dma",),
        )
        with pm.Model(coords={"dma": ["d0", "d1", "d2", "d3"]}):
            intercept.create_variable("intercept")  # dims ("dma",)
    """

    def __init__(
        self,
        mu: VariableFactory | dict | float | list[float],
        terms: dict[str, Term | dict],
        sigma: VariableFactory | dict,
        interactions: dict[str, Interaction | dict] | None = None,
        transform: str | None = "exp",
        centered: bool = False,
        dims: str | tuple[str, ...] | None = None,
    ) -> None:
        # Checks that need no model or dims: MMM.__init__ sets the dims, and no model exists yet
        if transform is not None:
            _get_transform(
                transform
            )  # the lookup Prior uses; raises UnknownTransformError
        if not isinstance(centered, bool):  # a quoted YAML "false" would be truthy
            raise ValueError(f"`centered` must be a bool, got {centered!r}")
        self.transform, self.centered = transform, centered
        self.terms: dict[str, Term] = {
            k: Term.coerce(k, t) for k, t in dict(terms).items()
        }
        self.interactions: dict[str, Interaction] = {
            k: Interaction.coerce(k, t) for k, t in dict(interactions or {}).items()
        }
        for kind, labels in (("term", self.terms), ("interaction", self.interactions)):
            if slashed := [k for k in labels if "/" in k]:  # part of variable names
                raise ValueError(
                    f"{kind} names {slashed} contain '/', which is not allowed in a variable name"
                )
        self.mu, self.sigma = _as_prior(mu), _as_prior(sigma)
        if not hasattr(
            self.mu, "create_variable"
        ):  # a constant: one number, or one per base label
            mu = np.asarray(self.mu)
            if mu.dtype.kind not in "iuf" or mu.ndim > 1:
                raise ValueError(
                    f"constant mu must be a number, or a list with one number per label "
                    f"of the base dim, got {self.mu!r}"
                )
            self.mu = (
                mu.tolist()
            )  # an ndarray as a list, so from_dict(to_dict()) == self
        if clash := sorted(set(self.terms) & set(self.interactions)):
            raise ValueError(
                f"interactions {clash} are named like terms; term and interaction names must be unique"
            )
        specs = [*self.terms, *self.interactions]
        sigma_dims = _as_dims(getattr(self.sigma, "dims", None))
        # Every option gets its level dim from create_variable; the options keyed by level are checked here
        levels = getattr(self.sigma, "levels", None)
        if (
            not isinstance(self.sigma, VariableFactory)
            or sigma_dims
            or (levels is not None and set(levels) != set(specs))
        ):
            raise ValueError(
                f"`sigma` must be IndependentSigmas or VarianceBudget keyed by exactly {sorted(specs)}, "
                f"or a prior with no dims, which gets the dim `<name>_sigma_level` (got dims {sigma_dims})"
            )
        if isinstance(self.sigma, Prior) and (
            vector := _non_scalar_params(self.sigma, "sigma")
        ):
            raise ValueError(
                f"{vector} hold more than one value: one value per level would be matched to the levels "
                "by position, i.e. by the order of `terms`. Use scalar parameters (each level gets its own draw), "
                "or IndependentSigmas / VarianceBudget, which are keyed by level"
            )
        parents = [t.within for t in self.terms.values() if t.within is not None]
        if shared := sorted(
            {q for q in parents if parents.count(q) > 1} & set(self.terms)
        ):
            raise ValueError(
                f"terms {shared} have more than one child `within` them; a parent term enters through one child only"
            )
        for c, t in self.terms.items():
            if t.within is None:
                continue
            if t.within in self.terms and self.terms[t.within].within is not None:
                raise ValueError(
                    f"term {c} is `within: {t.within}`, which is `within: {self.terms[t.within].within}`; "
                    "nesting is at most two levels per tree: a `within` parent cannot have a parent"
                )
            if t.within not in self.terms and t.parent_effect:
                raise ValueError(
                    f"term {c} is `within: {t.within}`, which is not a term: "
                    f"add {t.within} to `terms`, or set `parent_effect: false` on {c}"
                )
            if (
                t.within in self.terms and not t.parent_effect
            ):  # the parent would enter nowhere
                raise ValueError(
                    f"term {c} sets `parent_effect: false` but {t.within} is a term: "
                    f"drop {t.within} from `terms` and `sigma`, or drop `parent_effect`"
                )
        interaction_dims = [d for t in self.interactions.values() for d in t.dims]
        if not_coords := sorted(
            set(interaction_dims) - (set(self.terms) - set(parents))
        ):
            raise ValueError(
                f"interaction dims {not_coords} are not terms that are model coords (`within` parents are not coords)"
            )
        self.dims = dims

    @property
    def dims(self) -> tuple[str, ...] | None:
        """The dims of the coefficient; None until set (by the MMM, or the caller)."""
        return self._dims

    @dims.setter
    def dims(self, dims: str | tuple[str, ...] | None) -> None:
        self._dims = None if dims is None else _as_dims(dims)
        if self._dims is None:
            return
        parents = {t.within for t in self.terms.values()}
        hierarchy = {k for k in self.terms if k not in parents}
        hierarchy |= {d for t in self.interactions.values() for d in t.dims}
        base = [d for d in self._dims if d not in self.terms]
        if set(self._dims) - set(base) != hierarchy or len(base) > 1:
            raise ValueError(
                f"dims {self._dims} must be the hierarchy dims {sorted(hierarchy)} (the model-coord terms "
                f"and the interaction dims), in any order, and at most one base dim, got base dims {base}. "
                "For a hierarchy over only some of the MMM dims, pass `dims` explicitly"
            )
        is_prior = hasattr(self.mu, "create_variable")
        mu_dims = _as_dims(getattr(self.mu, "dims", None)) if is_prior else ()
        if mu_dims != tuple(base) if is_prior else np.ndim(self.mu) > len(base):
            raise ValueError(
                f"mu must have the base dims {tuple(base)}: a prior with those dims, a number, "
                f"or one number per base label; got {self.mu!r}"
            )

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
        if self.dims is None:
            raise ValueError(
                f"{name}: `dims` is unset; pass `dims`, or use the prior in an MMM, which sets it"
            )
        mu, terms, interactions = self.mu, self.terms, self.interactions
        model = pm.modelcontext(None)
        coords = model.coords
        specs = {**terms, **interactions}
        base = tuple(d for d in self.dims if d not in terms)
        if missing := [d for d in base if d not in coords]:
            raise ValueError(
                f"the model has no `{missing[0]}` coord; coords: {sorted(coords)}"
            )
        n_base = tuple(len(coords[d]) for d in base)
        # sorted, not in spec order: stored draws on the level dim are replayed by position, and equal priors
        # (whose terms may come back reordered, e.g. from YAML) must give the same order
        levels, level_dim = sorted(specs), f"{name}_sigma_level"
        # Data checks before any variable is created
        for c, t in terms.items():
            if c not in coords and not any(u.within == c for u in terms.values()):
                raise ValueError(
                    f"term {c}: not a model coord {sorted(coords)} and no term is `within` it"
                )
            # A term with a map (so `within`) is a model coord: the check above, as no term is `within` it (__init__)
            if t.map is None:
                continue
            if missing := [lab for lab in coords[c] if str(lab) not in t.map]:
                raise ValueError(
                    f"term {c}: `map` has no {t.within} for data labels {missing}"
                )
            if extra := sorted(set(t.map) - {str(lab) for lab in coords[c]}):
                raise ValueError(
                    f"term {c}: `map` keys {extra} are not {c} labels in the data {list(coords[c])}"
                )
        # Per nested term: the parent labels, and the parent index of each child in the coord order
        groups = {
            c: np.unique([t.map[str(lab)] for lab in coords[c]], return_inverse=True)
            for c, t in terms.items()
            if t.map is not None
        }
        if not hasattr(mu, "create_variable") and np.shape(mu) not in ((), n_base):
            raise ValueError(
                f"constant mu has shape {np.shape(mu)}; expected () or {n_base}, one value per {base} label"
            )
        parents = {t.within for t in terms.values()}
        # The coords this prior adds: the levels, and each parent term's labels
        new_coords = [
            (level_dim, levels),
            *(
                (t.within, groups[c][0].tolist())
                for c, t in terms.items()
                if t.within in terms
            ),
        ]
        for k, labs in new_coords:
            if k in coords and not np.array_equal(labs, coords[k]):
                raise ValueError(
                    f"model coord {k!r} has labels {list(coords[k])}, but {name} needs the labels {labs} "
                    "(the sigma levels, sorted; or a `within` parent's labels)"
                )
        # The RV and coord names this prior generates, checked before any exists; names in mu and sigma are pymc's
        names = [
            name,
            *(f"{name}_{k}" for k in specs),
            *(f"{name}_{c}_raw" for c in groups),
            # a non-centered Prior creates `<name>_offset`: of the ZeroSumNormal, or of the raw Normal of a child term
            *(
                f"{name}_{c}_raw_offset" if c in groups else f"{name}_{c}_offset"
                for c, t in terms.items()
                if not (self.centered if t.centered is None else t.centered)
            ),
            *(
                f"{name}_{k}_offset"
                for k, x in interactions.items()
                if not (self.centered if x.centered is None else x.centered)
            ),
            f"{name}_sigma",
            *([f"{name}_mu"] if hasattr(mu, "create_variable") else []),
            *(k for k, _ in new_coords),
        ]
        if duplicates := sorted({r for r in names if names.count(r) > 1}):
            raise ValueError(
                f"RV or coord names {duplicates} are generated twice: rename a term, interaction or parent label"
            )
        for c, (_, codes) in groups.items():  # a scale that only multiplies zeros
            if np.bincount(codes).max() < 2:
                warnings.warn(
                    f"term {c}: every {terms[c].within} has one {c}, so all {c} offsets are 0 and its sigma is unused",
                    stacklevel=2,
                )
        one_label = [  # parent terms, then crossed terms
            *(
                t.within
                for c, t in terms.items()
                if c in groups and t.within in terms and len(groups[c][0]) == 1
            ),
            *(
                c
                for c, t in terms.items()
                if t.within is None and c not in parents and len(coords[c]) == 1
            ),
        ]
        for c in one_label:
            warnings.warn(
                f"term {c} has one label, so its effect is 0 and its sigma is unused",
                stacklevel=2,
            )
        for k, x in interactions.items():
            if one := [d for d in x.dims if len(coords[d]) == 1]:
                warnings.warn(
                    f"interaction {k}: dims {one} have one label, so its offsets are 0 and its sigma is unused",
                    stacklevel=2,
                )
        # A no-op when the labels match, e.g. when a lazily built R2D2 decomposition or another prior added it first
        for k, labs in new_coords:
            model.add_coord(k, labs)
        sigma = copy.deepcopy(self.sigma)  # every sigma option: a copy on the level dim
        sigma.dims = (level_dim,)
        sigma = sigma.create_variable(f"{name}_sigma", xdist=True)
        # A factory may ignore its dims; the dims it returned fix the length (the coord `level_dim`)
        if (got := getattr(sigma.type, "dims", None)) != (level_dim,):
            raise ValueError(
                f"`sigma` must return an xtensor on the dims {(level_dim,)}, "
                f"one scale per level {levels}; got dims {got}"
            )

        def offset_args(level):  # the scale and the parameterization of one level
            centered = (
                self.centered
                if specs[level].centered is None
                else specs[level].centered
            )
            return sigma.isel({level_dim: levels.index(level)}), centered

        def zero_sum(level, dims, core_dims):
            scale, centered = offset_args(level)
            return Prior(
                "ZeroSumNormal",
                sigma=scale,
                centered=centered,
                core_dims=core_dims,
                dims=dims,
            ).create_variable(f"{name}_{level}", xdist=True)

        eta = as_xtensor(  # beta before the transform; a constant mu is a float or one value per base label
            mu.create_variable(f"{name}_mu", xdist=True)
            if hasattr(mu, "create_variable")
            else pt.broadcast_to(pt.as_tensor(mu), n_base),
            dims=base,
        )
        for c, t in terms.items():
            if t.within is not None:
                codes = groups[c][1]
                scale, centered = offset_args(c)
                offset = _GroupedZeroSumOffset(
                    scale=scale,
                    centered=centered,
                    dims=(*base, c),
                    child_dim=c,
                    codes=codes,
                ).create_variable(f"{name}_{c}", xdist=True)
                if t.within in terms:  # otherwise parent_effect is False (__init__)
                    parent = zero_sum(t.within, (*base, t.within), (t.within,))
                    offset = offset + parent.isel({t.within: codes}).rename(
                        {t.within: c}
                    )
                eta = eta + offset
            elif c not in parents:
                eta = eta + zero_sum(c, (*base, c), (c,))
        for k, x in interactions.items():
            eta = eta + zero_sum(k, (*base, *x.dims), tuple(x.dims))
        beta = eta.transpose(*self.dims)
        if self.transform is not None:
            beta = _get_transform(self.transform, xdist=True)(beta)
        beta = pmd.Deterministic(name, beta)
        return beta if xdist else beta.values

    def children(
        self, term: str, parent: Any, labels: Iterable[Any] | None = None
    ) -> list:
        """Return the labels of ``term`` that are under one parent, from the ``map`` of the term.

        The offsets of a child term are one variable over all of its labels, so this selects the ones of a parent.

        Parameters
        ----------
        term : str
            A term with ``within``.
        parent : label
            A parent label, a value of the ``map`` of ``term``.
        labels : array-like, optional
            The labels of ``term`` to choose from, such as the coord of the draws.
            The result then has their type and order, so it can select from the draws whatever the label type.
            Without it, the result is the keys of the ``map``, which are strings.

        Returns
        -------
        list
            The labels of ``term`` under ``parent``.

        Raises
        ------
        ValueError
            If ``term`` is not a term or has no ``within``, or if ``parent`` is not one of its parents.
            Also if ``labels`` has a label that is not in the ``map``.

        Examples
        --------
        .. code-block:: python

            beta = NestedZeroSumPrior(
                mu=Prior("Normal", dims="channel"),
                terms={
                    "family": {},
                    "brand": {
                        "within": "family",
                        "map": {"b0": "f0", "b1": "f0", "b2": "f1"},
                    },
                },
                sigma=IndependentSigmas(
                    {lv: Prior("HalfNormal") for lv in ("family", "brand")}
                ),
                dims=("brand", "channel"),
            )
            beta.children("brand", "f0")  # ["b0", "b1"]

            # the offsets of one family, from the draws of a model that has the prior as ``beta``
            brand = idata.prior["brand"].values
            idata.prior["beta_brand"].sel(
                brand=beta.children("brand", "f0", labels=brand)
            )
        """
        if term not in self.terms:
            raise ValueError(f"unknown term {term!r}; terms: {list(self.terms)}")
        spec = self.terms[term]
        if spec.within is None or spec.map is None:
            raise ValueError(f"term {term} has no `within`, so it has no parents")
        if parent not in spec.map.values():
            parents = sorted(set(spec.map.values()), key=str)
            raise ValueError(
                f"term {term} has no parent {parent!r}; parents: {parents}"
            )
        if labels is None:
            return [child for child, p in spec.map.items() if p == parent]
        if missing := [str(lab) for lab in labels if str(lab) not in spec.map]:
            raise ValueError(f"term {term}: `map` has no labels {missing}")
        return [lab for lab in labels if spec.map[str(lab)] == parent]

    def to_dict(self) -> dict:
        """Serialize to a dictionary of plain values (terms and interactions as plain dicts)."""
        return {
            "mu": self.mu.to_dict() if hasattr(self.mu, "to_dict") else self.mu,
            "terms": {k: t.to_dict() for k, t in self.terms.items()},
            "interactions": {k: t.to_dict() for k, t in self.interactions.items()},
            "sigma": self.sigma.to_dict()
            if hasattr(self.sigma, "to_dict")
            else self.sigma,
            "transform": self.transform,
            "centered": self.centered,
            "dims": None if self.dims is None else list(self.dims),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NestedZeroSumPrior":
        """Create the prior from a dictionary created by :meth:`to_dict`."""
        return cls(**{k: v for k, v in data.items() if k != "__type__"})

    def __repr__(self) -> str:
        """Return the call that creates the prior, without the arguments at their defaults."""
        args = [
            f"mu={self.mu!r}",
            f"terms={ {k: t.to_dict() for k, t in self.terms.items()}!r}",
        ]
        if self.interactions:
            interactions = {k: t.to_dict() for k, t in self.interactions.items()}
            args.append(f"interactions={interactions!r}")
        args.append(f"sigma={self.sigma!r}")
        if self.transform != "exp":
            args.append(f"transform={self.transform!r}")
        if self.centered:
            args.append("centered=True")
        if self.dims is not None:
            args.append(f"dims={self.dims!r}")
        return f"NestedZeroSumPrior({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        """Equal when the serialized forms are: the order of map, sigma and spec keys does not matter."""
        return (
            isinstance(other, NestedZeroSumPrior) and self.to_dict() == other.to_dict()
        )

    def __hash__(self) -> int:
        """Hash the parts :meth:`__eq__` compares that are order-free, so equal priors hash equal."""
        return hash(
            (
                type(self).__name__,
                self.dims,
                self.transform,
                self.centered,
                frozenset(self.terms),
                frozenset(self.interactions),
            )
        )


# MMM.load reads the saturation priors back through pymc_extras.deserialize, which does not consult
# pymc-marketing's registry, so the classes are also found there by `__type__`.
for _cls in (IndependentSigmas, VarianceBudget, NestedZeroSumPrior):
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
