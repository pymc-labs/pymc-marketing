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
"""Base classes containing primitives and high-level API for model building, fitting, saving, and loading."""

import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Literal, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace
from pymc.model.core import Model
from pymc.util import RandomState
from pymc.variational.callbacks import CheckParametersConvergence
from pymc_extras.deserialize import DeserializableError, deserialize
from pymc_extras.printing import model_table
from rich.table import Table

from pymc_marketing.data.idata.utils import idata_from_zarr, idata_to_zarr
from pymc_marketing.version import __version__

# If scikit-learn is available, use its data validator
try:
    from sklearn.utils.validation import check_array, check_X_y
# If scikit-learn is not available, return the data unchanged
except ImportError:

    def check_X_y(X, y, **kwargs):
        """Check if the input data is valid for the model."""
        return X, y

    def check_array(X, **kwargs):
        """Check if the input data is valid for the model."""
        return X


def create_idata_accessor(value: str, message: str):
    """Create a property accessor for a group of the model's DataTree.

    Underlying object must have a ``xr.DataTree`` attribute named 'idata'.

    Parameters
    ----------
    value : str
        The group to access in the DataTree.
    message : str
        The error message to raise if the group is not found in the DataTree.

    Returns
    -------
    property
        The property accessor for the DataTree group.

    """

    def accessor(self) -> xr.Dataset:
        if self.idata is None or value not in self.idata.children:
            raise RuntimeError(message)

        return self.idata[f"/{value}"].to_dataset()

    return property(
        accessor,
        doc=f"Access the '{value}' group of the DataTree.",
    )


def requires_model(func):
    """Ensure that the model is built before calling a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "model"):
            raise RuntimeError(
                "The model hasn't been built yet. Please call `build_model` first."
            )
        return func(self, *args, **kwargs)

    return wrapper


def create_sample_kwargs(
    sampler_config: dict[str, Any],
    progressbar: bool | None,
    random_seed,
    **kwargs,
) -> dict[str, Any]:
    """Create the dictionary of keyword arguments for `pm.sample`.

    Parameters
    ----------
    sampler_config : dict
        The configuration dictionary for the sampler.
    progressbar : bool, optional
        Whether to show the progress bar during sampling. Defaults to True.
    random_seed : RandomState
        The random seed for the sampler.
    **kwargs : Any
        Additional keyword arguments to pass to the sampler.

    Returns
    -------
    dict
        The dictionary of keyword arguments for `pm.sample`.

    """
    sampler_config = sampler_config.copy()

    if progressbar is not None:
        sampler_config["progressbar"] = progressbar
    else:
        sampler_config["progressbar"] = sampler_config.get("progressbar", True)

    if random_seed is not None:
        sampler_config["random_seed"] = random_seed

    sampler_config.update(**kwargs)

    return sampler_config


class DifferentModelError(Exception):
    """Error raised when a model loaded is different than one saved."""


class ModelIO:
    """Mixin to handle saving and loading of models."""

    _model_type: str
    version: str
    idata: xr.DataTree | None
    sampler_config: dict
    model_config: dict

    @property
    def id(self) -> str:
        """Generate a unique hash value for the model.

        The hash value is created using the last 16 characters of the SHA256 hash encoding,
        based on the model configuration, version, and model type.

        Returns
        -------
        str
            A string of length 16 characters containing a unique hash of the model.

        Examples
        --------
        .. code-block:: python

            model = MyModel()
            model.id
            "0123456789abcdef"

        """

        def _serialize_for_hash(obj):
            """Serialize objects for deterministic hashing."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "model_dump"):
                # Handle Pydantic models (e.g., HSGPKwargs)
                return obj.model_dump(mode="json")
            # For other objects, try to use their __dict__ but filter out non-serializable items
            if hasattr(obj, "__dict__"):
                # Filter out methods, functions, and other non-serializable attributes
                return {
                    k: v
                    for k, v in obj.__dict__.items()
                    if not callable(v) and not k.startswith("_")
                }
            # Last resort: convert to string representation
            return str(obj)

        hasher = hashlib.sha256()
        # Use JSON serialization with custom default for deterministic model IDs
        config_json = json.dumps(
            self._serializable_model_config, sort_keys=True, default=_serialize_for_hash
        )
        hasher.update(config_json.encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]

    @property
    @abstractmethod
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        """Converts non-serializable values from model_config to their serializable reversable equivalent.

        Data types like pandas DataFrame, Series or datetime aren't JSON serializable,
        so in order to save the model they need to be formatted.

        Returns
        -------
        model_config : dict

        """

    def create_idata_attrs(self) -> dict[str, str]:
        """Create attributes for the inference data.

        Returns
        -------
        dict[str, str]
            A dictionary of attributes for the inference data.
        """

        def _json_default(obj):
            """Handle objects that aren't JSON serializable by default."""
            from pymc_marketing.serialization import serialization

            try:
                return serialization.serialize(obj)
            except (KeyError, TypeError):
                pass

            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json")
            if hasattr(obj, "__dict__"):
                return {
                    k: v
                    for k, v in obj.__dict__.items()
                    if not callable(v) and not k.startswith("_")
                }
            return str(obj)

        attrs: dict[str, str] = {}

        attrs["id"] = self.id
        attrs["model_type"] = self._model_type
        attrs["version"] = self.version
        attrs["sampler_config"] = json.dumps(self.sampler_config, default=_json_default)
        attrs["model_config"] = json.dumps(
            self._serializable_model_config, default=_json_default
        )

        return attrs

    def set_idata_attrs(self, idata: xr.DataTree | None = None) -> xr.DataTree:
        """Set attributes on a DataTree object.

        Parameters
        ----------
        idata : xr.DataTree, optional
            The DataTree object to set attributes on.

        Raises
        ------
        ValueError
            If the attrs are missing for a property initialization of the class
        RuntimeError
            If no DataTree object is provided.

        Returns
        -------
        DataTree
            The DataTree instance with the attrs set

        Examples
        --------
        Set the attrs for a DataTree object manually.

        .. code-block:: python

            idata: xr.DataTree = ...
            model.set_idata_attrs(idata=idata)

        """
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")

        attrs = self.create_idata_attrs()
        attrs_keys = set(attrs.keys())
        required_keys = {
            "id",
            "model_type",
            "version",
            "sampler_config",
            "model_config",
        }
        if missing_keys := required_keys - attrs_keys:
            msg = (
                f"Missing required keys in attrs: {missing_keys}. "
                "Call super().create_idata_attrs()."
            )
            raise ValueError(msg)

        init_parameters: set[str] = set(signature(self.__init__).parameters.keys())  # type: ignore
        # Remove data attr since it will be stored in the fit_data group of DataTree
        init_parameters -= {"data"}

        if missing_keys := init_parameters - attrs_keys:
            msg = (
                f"__init__ has parameters that are not in the attrs: {missing_keys}. "
                "The save and load functionality will not work correctly."
            )
            raise ValueError(msg)

        idata.attrs = attrs
        return idata

    def save(self, fname: str, **kwargs) -> None:
        """Save the model's inference data to a file.

        Parameters
        ----------
        fname : str
            The name and path of the file to save the inference data with model parameters.
        **kwargs
            Additional keyword arguments to pass to xr.DataTree.to_netcdf().
            Common options include:

            - ``engine`` : str, optional (default ``"netcdf4"``)
              Library to use for writing files.
            - ``groups`` : list of str, optional
              Groups to save to netcdf. If None, all groups are saved.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model hasn't been fit yet (no inference data available).

        Examples
        --------
        This method is meant to be overridden and implemented by subclasses.
        It should not be called directly on the base abstract class or its instances.

        .. code-block:: python

            class MyModel(ModelBuilder):
                def __init__(self):
                    super().__init__()


            model = MyModel()
            model.fit(X, y)
            # Basic save
            model.save("model_results.nc")

            # Save with specific options
            model.save(
                "model_results.nc",
                engine="netcdf4",
                groups=["posterior", "log_likelihood"],
            )

        """
        if self.idata is not None and "posterior" in self.idata:
            file = Path(str(fname))
            groups = kwargs.pop("groups", None)
            idata_to_save = self.idata
            if groups is not None:
                groups_with_slash = {
                    g if g.startswith("/") else f"/{g}" for g in groups
                }
                nodes_to_drop = [
                    node[1:]
                    for node in self.idata.groups
                    if node != "/" and node not in groups_with_slash
                ]
                idata_to_save = self.idata.drop_nodes(nodes_to_drop)
            if file.suffix == ".zarr" or file.is_dir():
                idata_to_zarr(idata_to_save, file, **kwargs)
            else:
                idata_to_save.to_netcdf(str(file), **kwargs)
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """Format the model configuration.

        Recursively processes the config dict.  Dicts with a ``__type__`` key
        are deserialized via the TypeRegistry.  Prior specs (``dist``) and
        wrapper factories that are not registered in the TypeRegistry
        (e.g. ``Censored``, which serializes to a ``class``/``data`` pair) are
        rebuilt via the pymc-extras deserializer.  Plain lists are converted
        back to tuples (for ``dims``) or numpy arrays (everything else) to undo
        the JSON round-trip.
        """
        from pymc_marketing.serialization import serialization

        def _looks_like_prior_spec(value: Any) -> bool:
            return isinstance(value, dict) and (
                isinstance(value.get("dist"), str)
                or (isinstance(value.get("class"), str) and "data" in value)
            )

        def _format(d: dict) -> dict:
            for key, value in d.items():
                if isinstance(value, dict) and "__type__" in value:
                    d[key] = serialization.deserialize(value)
                # Must precede the generic dict branch: recursing first would
                # rebuild the nested ``dist`` and leave the wrapper unreadable.
                elif _looks_like_prior_spec(value):
                    try:
                        d[key] = deserialize(value)
                    except DeserializableError as err:
                        # ``deserialize`` raises this for two different
                        # situations. With ``__cause__`` unset, no registered
                        # deserializer matched, so this is an unrelated config
                        # mapping that merely uses these keys: recurse rather
                        # than making the whole model unloadable. With
                        # ``__cause__`` set, a deserializer did match and its
                        # ``from_dict`` failed, which is a real error and must
                        # not be silently downgraded to a raw dict.
                        if err.__cause__ is not None:
                            raise
                        d[key] = _format(value)
                elif isinstance(value, dict):
                    d[key] = _format(value)
                elif isinstance(value, list):
                    if key == "dims":
                        d[key] = tuple(value)
                    elif value and all(isinstance(v, (int, float)) for v in value):
                        d[key] = np.array(value)
            return d

        return _format(model_config.copy())

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        """Convert the model configuration and sampler configuration from the attributes to keyword arguments.

        This method must be overridden in child classes if additional keyword arguments are needed.
        """
        return {
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "sampler_config": json.loads(attrs["sampler_config"]),
        }

    @classmethod
    def idata_to_init_kwargs(cls, idata: xr.DataTree) -> dict[str, Any]:
        """Create  the model configuration and sampler configuration from the DataTree to keyword arguments.

        This method must be overridden in child classes to add additional keyword arguments.
        """
        return cls.attrs_to_init_kwargs(idata.attrs)

    @abstractmethod
    def build_from_idata(self, idata: xr.DataTree) -> None:
        """Build the model from the DataTree object."""

    @classmethod
    def load(cls, fname: str, check: bool = True):
        """Create a ModelBuilder instance from a file.

        Loads inference data for the model.

        This class method has a few steps:

        - Load the DataTree from the file.
        - Construct a new instance of the model using the DataTree attrs
        - Build the model from the DataTree
        - Check if the model id matches the id in the DataTree loaded.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.
        check : bool, optional
            Whether to check if the model id matches the id in the DataTree loaded.
            Defaults to True.

        Returns
        -------
        Returns an instance of ModelBuilder.

        Raises
        ------
        DifferentModelError
            If the inference data that is loaded doesn't match with the model.

        Examples
        --------
        Load a model from a file

        .. code-block:: python

            file_name: str = "./mymodel.nc"
            model = MyModel.load(file_name)

        """
        filepath = Path(str(fname))
        if filepath.suffix == ".zarr" or filepath.is_dir():
            idata = idata_from_zarr(filepath)
        else:
            idata = xr.open_datatree(str(filepath))

        try:
            return cls.load_from_idata(idata, check=check)
        except DifferentModelError as e:
            error_msg = (
                f"The file '{fname}' does not contain "
                "a DataTree of the same model "
                f"or configuration as '{cls._model_type}'"
            )
            raise DifferentModelError(error_msg) from e

    @classmethod
    def load_from_idata(cls, idata: xr.DataTree, check: bool = True) -> "ModelIO":
        """Create a ModelBuilder instance from a DataTree object.

        This class method has a few steps:

        - Construct a new instance of the model using the DataTree attrs
        - Build the model from the DataTree
        - Check if the model id matches the id in the DataTree loaded.

        Parameters
        ----------
        idata : xr.DataTree
            The DataTree object to load the model from.
        check : bool, optional
            Whether to check if the model id matches the id in the DataTree loaded.
            Defaults to True.

        Returns
        -------
        ModelBuilder
            An instance of the ModelBuilder class.

        Raises
        ------
        DifferentModelError
            If the model id in the DataTree does not match the model id built.

        """
        init_kwargs = cls.idata_to_init_kwargs(idata)

        model = cls(**init_kwargs)

        model.idata = idata
        if "fit_data" in idata:
            # TODO: Overriding method in CLVModel requires this; revise/remove for v1.0
            built = model.build_from_idata(idata)
            if built is not None:
                model = built
        else:
            warnings.warn(
                "The loaded model does not include fit_data used for training. "
                "Plotting and prior/posterior predictive sampling may not work correctly. "
                "Run build_model() with training data for full functionality.",
                UserWarning,
                stacklevel=2,
            )

        if not check:
            return model

        if (model_version := model.version) != (
            loaded_version := idata.attrs["version"]
        ):
            msg = (
                f"The model version ({loaded_version}) in the DataTree does not "
                f"match the model version ({model_version}). "
                "There was no error loading the inference data, but the model structure "
                "is different. "
            )
            raise DifferentModelError(msg)

        if model.id != idata.attrs["id"]:
            msg = (
                "The model id in the DataTree does not match the model id. "
                "There was no error loading the inference data, but the model may "
                "be different. "
                "Investigate if the model structure or configuration has changed."
            )
            raise DifferentModelError(msg)

        return model


#: Sampling methods supported by :meth:`ModelFitter.fit`.
SamplingMethod = Literal["mcmc", "map", "demz", "advi", "fullrank_advi"]

_APPROX_METHODS: tuple[str, ...] = ("advi", "fullrank_advi")

#: ``fit`` keyword arguments that belong to ``Approximation.sample`` rather than ``pymc.fit``.
_APPROX_SAMPLE_KEYS: tuple[str, ...] = ("draws", "return_inferencedata")

#: Keys that only apply when ``pm.sample`` picks its own NUTS stepper. Passing them
#: alongside an explicit ``step`` makes ``pm.sample`` raise, so the gradient-free path
#: has to drop them.
_NUTS_ONLY_KEYS: tuple[str, ...] = ("target_accept", "nuts", "nuts_sampler", "init")

#: Groups derived from a particular posterior. Dropped before merging a new fit into an
#: existing ``idata`` so a refit cannot leave e.g. MCMC ``sample_stats`` attached to a
#: MAP posterior. ``prior``/``prior_predictive`` are posterior-independent and survive.
_POSTERIOR_DERIVED_GROUPS: tuple[str, ...] = (
    "sample_stats",
    "log_likelihood",
    "posterior_predictive",
    "predictions",
    "warmup_posterior",
    "warmup_sample_stats",
)


def _approx_fit_parameters() -> set[str]:
    """Collect the keyword arguments accepted by the variational fitting stack.

    ``pymc.fit`` forwards ``**kwargs`` down to ``Inference.fit`` and from there to
    ``ObjectiveFunction.step_function``, so the accepted names are spread across three
    signatures. Deriving them keeps the filter correct as PyMC evolves, instead of
    hard-coding a list that silently goes stale.

    Returns
    -------
    set of str
        Names accepted somewhere in the ``pymc.fit`` call chain.

    """
    from pymc.variational.inference import Inference
    from pymc.variational.opvi import ObjectiveFunction

    names: set[str] = set()
    for func in (pm.fit, Inference.fit, ObjectiveFunction.step_function):
        names.update(
            name
            for name, param in signature(func).parameters.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            and name != "self"
        )
    return names


def _map_incompatible_parameters() -> set[str]:
    """Collect ``pm.sample`` parameter names that ``pm.find_MAP`` does not accept.

    ``find_MAP`` forwards unknown keyword arguments to ``scipy.optimize.minimize``,
    so an MCMC key such as ``draws`` surfaces as an opaque scipy ``TypeError``.
    Deriving the incompatible names from the two signatures lets the MAP path strip
    them with a clear warning instead.

    Returns
    -------
    set of str
        Names accepted by ``pm.sample`` but not by ``pm.find_MAP``.

    """

    def named(func: Callable) -> set[str]:
        return {
            name
            for name, param in signature(func).parameters.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        }

    return named(pm.sample) - named(pm.find_MAP)


def _normalize_map_seed(random_seed: RandomState) -> int | None:
    """Convert a ``fit(random_seed=...)`` value into the int ``pm.find_MAP`` accepts.

    Returns
    -------
    int or None
        An integer seed, or ``None`` (with a warning) when the value cannot be
        converted -- so unsupported seed types are surfaced instead of silently
        leaving the MAP optimization unseeded.

    """
    if isinstance(random_seed, (int, np.integer)):
        return int(random_seed)
    if isinstance(random_seed, np.random.Generator):
        return int(random_seed.integers(2**32))
    if isinstance(random_seed, np.random.RandomState):
        return int(random_seed.randint(2**31))
    warnings.warn(
        f"random_seed of type {type(random_seed).__name__} is not supported with "
        "method='map' and was ignored; pass an int or numpy Generator instead.",
        UserWarning,
        stacklevel=3,
    )
    return None


class ModelFitter:
    """Mixin providing a unified fitting API for all PyMC-Marketing models.

    Owns the whole fit pipeline: model preparation, sampler dispatch, deterministic
    recomputation, ``idata`` merging, and the ``fit_data`` group. Subclasses customize
    behaviour through the hooks below rather than by reimplementing :meth:`fit`.

    Hooks
    -----
    _prepare_fit
        Resolve input data and ensure ``self.model`` exists.
    create_fit_data_group
        Build the ``fit_data`` group, or return ``None`` to omit it.
    _get_sampling_model
        Return the model actually handed to the sampler (e.g. with frozen dims).
    post_sample_model_transformation
        Run after sampling, before the ``idata`` is assembled.

    Examples
    --------
    .. code-block:: python

        class MyModel(ModelBuilder): ...


        model = MyModel()
        idata = model.fit(data, method="mcmc")

    """

    # Attributes supplied by the host class.
    model: pm.Model
    idata: xr.DataTree | None
    sampler_config: dict
    is_fitted_: bool

    #: When True, sample the free RVs only and recompute deterministics vectorized
    #: afterwards. Set to False on models whose deterministics are large enough that the
    #: vectorized recompute would spike transient memory.
    _recompute_deterministics: bool = True

    def _prepare_fit(self, data: Any = None) -> None:
        """Resolve the input data and make sure a model is built.

        Parameters
        ----------
        data : Any, optional
            Data passed to :meth:`fit`. ``None`` means "use whatever the instance
            already holds". The base implementation cannot route data anywhere, so
            it raises rather than silently fitting stale data; subclasses that accept
            data through :meth:`fit` must override this hook.

        """
        if data is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not accept `data` through `fit()`; "
                "override `_prepare_fit` to support it."
            )
        if not hasattr(self, "model"):
            self.build_model()  # type: ignore[attr-defined]

    def create_fit_data_group(self) -> xr.Dataset | None:
        """Build the ``fit_data`` group stored alongside the posterior.

        Returns
        -------
        xr.Dataset or None
            The training data as a Dataset, or ``None`` to omit the group entirely.

        """
        data = getattr(self, "data", None)
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data.to_xarray()
        return data

    def _get_sampling_model(self) -> pm.Model:
        """Return the model handed to the sampler."""
        return self.model

    def post_sample_model_transformation(self) -> None:
        """Perform transformation on the model after sampling."""

    def _sample_with_deterministics(
        self,
        sampler_kwargs: dict[str, Any],
        step_factory: Callable[[], Any] | None = None,
    ) -> xr.DataTree:
        """Sample, optionally deferring deterministics to a vectorized recompute.

        Computing deterministics inside the sampling loop is wasteful for models with
        large deterministic nodes, and is redundant work for the JAX/nutpie backends.
        Sampling the free RVs alone and recomputing afterwards yields identical values.

        Parameters
        ----------
        sampler_kwargs : dict
            Keyword arguments for ``pm.sample``.
        step_factory : callable, optional
            Builds the step method. Called inside the sampling model's context so the
            step is bound to the same model instance that is sampled -- overrides of
            ``_get_sampling_model`` may return a fresh object on every call.

        """
        model = self._get_sampling_model()
        with model:
            kwargs = dict(sampler_kwargs)
            if step_factory is not None:
                kwargs["step"] = step_factory()

            if not self._recompute_deterministics or not model.deterministics:
                return pm.sample(**kwargs)

            var_names = [var.name for var in model.free_RVs]
            idata = pm.sample(var_names=var_names, **kwargs)
            idata["/posterior"] = pm.compute_deterministics(
                idata["/posterior"], merge_dataset=True
            )
        return idata

    def _fit_mcmc(self, sampler_kwargs: dict[str, Any]) -> xr.DataTree:
        """Fit a model with NUTS."""
        return self._sample_with_deterministics(sampler_kwargs)

    def _fit_demz(self, sampler_kwargs: dict[str, Any]) -> xr.DataTree:
        """Fit a model with the DEMetropolisZ gradient-free sampler."""
        kwargs = {k: v for k, v in sampler_kwargs.items() if k not in _NUTS_ONLY_KEYS}

        if removed := sorted(set(sampler_kwargs) - set(kwargs)):
            warnings.warn(
                "The following keyword arguments only apply to NUTS and were "
                f"removed before sampling with 'demz': {removed}.",
                UserWarning,
                stacklevel=2,
            )

        return self._sample_with_deterministics(kwargs, step_factory=pm.DEMetropolisZ)

    def _fit_map(self, **kwargs: Any) -> xr.DataTree:
        """Find the model maximum a posteriori using a scipy optimizer."""
        if removed := sorted(set(kwargs) & _map_incompatible_parameters()):
            warnings.warn(
                "The following keyword arguments only apply to MCMC sampling and "
                f"were removed before optimizing with 'map': {removed}.",
                UserWarning,
                stacklevel=2,
            )
            for key in removed:
                kwargs.pop(key)

        model = self._get_sampling_model()
        map_res = pm.find_MAP(model=model, **kwargs)
        # Filter non-value variables
        value_vars_names = set(v.name for v in cast(Model, model).value_vars)
        map_res = {k: v for k, v in map_res.items() if k in value_vars_names}
        # Convert map result to DataTree
        map_strace = NDArray(model=model)
        map_strace.setup(draws=1, chain=0)
        try:
            map_strace.record(map_res, in_warmup=False)
        except TypeError:
            map_strace.record(map_res)
        map_strace.close()
        trace = MultiTrace([map_strace])
        return pm.to_inference_data(trace, model=model)

    def _fit_approx(
        self,
        method: str = "advi",
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, xr.DataTree]:
        """Fit a model with variational inference and draw from the approximation.

        Keyword arguments are split between ``pymc.fit`` and ``Approximation.sample``.
        The accepted names are derived from the ``pymc.fit`` call chain rather than
        hard-coded, and anything the variational stack cannot accept is reported in a
        ``UserWarning`` instead of being dropped silently. This matters because
        ``sampler_config`` is shared with the MCMC path and legitimately carries
        MCMC-only keys such as ``tune`` and ``chains``.
        """
        config = self.sampler_config.copy() if self.sampler_config else {}
        merged = {**config, **kwargs}

        if merged.get("method") is not None:
            raise ValueError(
                "The 'method' parameter is set in sampler_config. "
                f"Cannot be called with '{method}'."
            )
        if merged.get("chains", 1) > 1:
            warnings.warn(
                f"The 'chains' parameter must be 1 with '{method}'. "
                "Sampling only 1 chain despite the provided parameter.",
                UserWarning,
                stacklevel=2,
            )

        _sample_kwargs: dict[str, Any] = {
            k: merged[k] for k in _APPROX_SAMPLE_KEYS if k in merged
        }
        _sample_kwargs.update(sample_kwargs or {})

        allowed = _approx_fit_parameters()
        fit_kwargs = {
            k: v
            for k, v in merged.items()
            if k in allowed and k not in _APPROX_SAMPLE_KEYS
        }

        # "chains" already has its own dedicated warning above.
        if ignored := sorted(
            set(merged)
            - set(fit_kwargs)
            - set(_APPROX_SAMPLE_KEYS)
            - {"method", "chains"}
        ):
            warnings.warn(
                f"The following keyword arguments are not accepted by '{method}' "
                f"and will be ignored: {ignored}.",
                UserWarning,
                stacklevel=2,
            )

        if progressbar is not None:
            fit_kwargs["progressbar"] = progressbar
        if random_seed is not None:
            fit_kwargs["random_seed"] = random_seed
        # A seed supplied through sampler_config must make the approximation draws
        # reproducible too, not just the optimization.
        seed = random_seed if random_seed is not None else merged.get("random_seed")
        if seed is not None:
            _sample_kwargs.setdefault("random_seed", seed)

        _sample_kwargs.setdefault("draws", 1_000)
        fit_kwargs.setdefault(
            "callbacks", [CheckParametersConvergence(diff="absolute")]
        )

        with self._get_sampling_model():
            approx = pm.fit(method=method, **fit_kwargs)
            return approx, approx.sample(**_sample_kwargs)

    def fit(
        self,
        data: Any = None,
        *,
        method: SamplingMethod = "mcmc",
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> xr.DataTree:
        """Infer the model posterior.

        Sets attrs on the inference data of the model.

        Parameters
        ----------
        data : Any, optional
            Input data for model fitting. If ``None``, the data already held by the
            instance is used. Models that do not override ``_prepare_fit`` raise a
            ``NotImplementedError`` when data is passed here.
        method : str
            Method used to fit the model. Options are:

            - ``"mcmc"``: Samples from the posterior via `pymc.sample` (default)
            - ``"map"``: Finds maximum a posteriori via `pymc.find_MAP`
            - ``"demz"``: Samples from the posterior via `pymc.sample` using DEMetropolisZ
            - ``"advi"``: Samples via `pymc.fit(method="advi")` and `Approximation.sample`
            - ``"fullrank_advi"``: As ``"advi"``, with a full-rank approximation

        progressbar : bool, optional
            Specifies whether the fit progress bar should be displayed. Defaults to True.
        random_seed : RandomState, optional
            Provides the sampler with an initial random seed for reproducible samples.
        sample_kwargs : dict, optional
            Only used by the variational methods; forwarded to ``Approximation.sample``
            (e.g. ``{"draws": 1_000}``).
        **kwargs : Any
            Custom sampler settings, passed to the underlying PyMC routine.

        Returns
        -------
        xr.DataTree
            Inference data of the fitted model.

        Examples
        --------
        .. code-block:: python

            model = MyModel()
            idata = model.fit(data)
            idata = model.fit(data, method="map")

        """
        self._prepare_fit(data)

        def sampler_kwargs() -> dict[str, Any]:
            return create_sample_kwargs(
                self.sampler_config,
                progressbar,
                random_seed,
                **kwargs,
            )

        approx = None
        match method:
            case "mcmc":
                idata = self._fit_mcmc(sampler_kwargs())
            case "demz":
                idata = self._fit_demz(sampler_kwargs())
            case "map":
                map_kwargs = dict(kwargs)
                if progressbar is not None:
                    map_kwargs.setdefault("progressbar", progressbar)
                if random_seed is not None:
                    if (seed := _normalize_map_seed(random_seed)) is not None:
                        map_kwargs.setdefault("seed", seed)
                idata = self._fit_map(**map_kwargs)
            case method if method in _APPROX_METHODS:
                approx, idata = self._fit_approx(
                    method=method,
                    progressbar=progressbar,
                    random_seed=random_seed,
                    sample_kwargs=sample_kwargs,
                    **kwargs,
                )
            case _:
                raise ValueError(
                    "Fit method options are ['mcmc', 'map', 'demz', 'advi', "
                    f"'fullrank_advi'], got: {method}"
                )

        if approx is not None:
            self.approx = approx

        self.post_sample_model_transformation()

        if self.idata:
            self.idata = self.idata.copy()
            if stale := [
                group
                for group in _POSTERIOR_DERIVED_GROUPS
                if group in self.idata.children
            ]:
                self.idata = self.idata.drop_nodes(stale)
            self.idata.update(idata)
        else:
            self.idata = idata

        self.idata["/posterior"].attrs["pymc_marketing_version"] = __version__

        if "fit_data" in self.idata.children:
            self.idata = self.idata.drop_nodes("fit_data")

        fit_data = self.create_fit_data_group()
        if fit_data is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="The group fit_data is not defined in the DataTree scheme",
                )
                self.idata["/fit_data"] = fit_data

        self.set_idata_attrs(self.idata)  # type: ignore[attr-defined]
        self.is_fitted_ = True
        return self.idata


class ModelBuilder(ABC, ModelIO, ModelFitter):
    """Base class for building PyMC-Marketing models.

    Child classes must implement the following methods:
    - default_model_config: Returns a dictionary for default model configuration.
    - default_sampler_config: Returns a dictionary for default sampler configuration.
    - build_model: Builds the model based on the provided data and model configuration.
    - build_from_idata: Builds the model from a DataTree object. Needed for loading models.
    - fit: Fits the model based on the provided data and sampler configurations.
    - attrs_to_init_kwargs: Override to add additional init keyword arguments.
    - _serializable_model_config: Needed for saving and loading the model.

    """

    _model_type = "BaseClass"
    version = "None"
    _skipped_config_keys: set[str] = set()

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        """Initialize model configuration and sampler configuration for the model.

        Parameters
        ----------
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration.
            Class-default defined by the user default_model_config method.
            A ``UserWarning`` is raised for any key not present in
            ``default_model_config``, since such keys are ignored by the model.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration.
            Class-default defined by the user default_sampler_config method.

        Examples
        --------
        .. code-block:: python

            class MyModel(ModelBuilder): ...


            model = MyModel(model_config, sampler_config)

        """
        if sampler_config is None:
            sampler_config = {}
        if model_config is None:
            model_config = {}

        self.sampler_config = (
            self.default_sampler_config | sampler_config
        )  # Parameters for fit sampling
        default_model_config = self.default_model_config
        self.model_config = (
            default_model_config | model_config
        )  # parameters for priors etc.

        # Warn about model_config keys that the model does not use, so that
        # typos (e.g. "alphaa" instead of "alpha") don't silently get ignored.
        unused_model_config_keys = (
            set(model_config) - set(default_model_config) - self._skipped_config_keys
        )
        if unused_model_config_keys:
            warnings.warn(
                "The following model_config keys are not used by the model "
                f"and will be ignored: {sorted(unused_model_config_keys)}. "
                f"Valid keys are: {sorted(default_model_config)}.",
                UserWarning,
                stacklevel=2,
            )

        self.model: pm.Model
        self.idata: xr.DataTree | None = None  # idata is generated during fitting
        self.is_fitted_ = False

    @property
    @abstractmethod
    def default_model_config(self) -> dict:
        """Return a class default configuration dictionary.

        For model builder if no model_config is provided on class initialization
        Useful for understanding structure of required model_config to allow its customization by users

        Examples
        --------
        .. code-block:: python

            @classmethod
            def default_model_config(self):
                Return {
                    'a' : {
                        'loc': 7,
                        'scale' : 3
                    },
                    'b' : {
                        'loc': 3,
                        'scale': 5
                    },
                     'obs_error': 2
                }

        Returns
        -------
        model_config : dict
            A set of default parameters for predictor distributions that allow to save and recreate the model.

        """

    @property
    @abstractmethod
    def default_sampler_config(self) -> dict:
        """Return a class default sampler configuration dictionary.

        For model builder if no sampler_config is provided on class initialization
        Useful for understanding structure of required sampler_config to allow its customization by users

        Examples
        --------
        .. code-block:: python

            @classmethod
            def default_sampler_config(self):
                Return {
                    'draws': 1_000,
                    'tune': 1_000,
                    'chains': 1,
                    'target_accept': 0.95,
                }

        Returns
        -------
        sampler_config : dict
            A set of default settings for used by model in fit process.

        """

    @abstractmethod
    def build_model(
        self,
        **kwargs,
    ) -> None:
        """Create an instance of `pm.Model` based on provided data and model_config.

        It attaches the model to self.model.

        Parameters
        ----------
        kwargs : dict
            data arguments for model configuration.

        See Also
        --------
        default_model_config : returns default model config

        Returns
        -------
        None

        """

    @requires_model
    def graphviz(self, **kwargs):
        """Get the graphviz representation of the model.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the `pm.model_to_graphviz` function

        Returns
        -------
        graphviz.Digraph

        """
        return pm.model_to_graphviz(self.model, **kwargs)

    @requires_model
    def table(self, **model_table_kwargs) -> Table:
        """Get the summary table of the model.

        Parameters
        ----------
        **model_table_kwargs
            Keyword arguments for the `model_table` function

        Returns
        -------
        rich.table.Table
            A rich table containing the summary of the model.

        """
        return model_table(self.model, **model_table_kwargs)

    @property
    def fit_result(self) -> xr.Dataset:
        """Get the posterior fit_result.

        Returns
        -------
        DataTree object.

        """
        return create_idata_accessor(
            "posterior", "The model hasn't been fit yet, call .fit() first"
        ).__get__(self)

    @fit_result.setter
    def fit_result(self, res: xr.DataTree) -> None:
        """Create a setter method to overwrite the pre-existing fit_result.

        Parameters
        ----------
        res : xr.DataTree
            The DataTree object to be set

        """
        if self.idata is None:
            self.idata = res
        elif "posterior" in self.idata.children:
            warnings.warn("Overriding pre-existing fit_result", stacklevel=2)
            self.idata["/posterior"] = res["/posterior"].to_dataset()
        else:
            if "posterior" in res.children:
                self.idata["/posterior"] = res["/posterior"].to_dataset()
            else:
                # ``res`` has no explicit posterior group (e.g. a flat DataTree
                # built from a bare Dataset): flatten every group's variables
                # into a single posterior Dataset, keeping the first occurrence
                # of each variable name.
                posterior_flat = xr.Dataset()
                for g in res.groups:
                    if g == "/":
                        continue
                    for var_name, var in res[g].dataset.variables.items():
                        if var_name not in posterior_flat:
                            posterior_flat[var_name] = var
                self.idata["/posterior"] = posterior_flat

    prior = create_idata_accessor(
        "prior",
        "The model hasn't been sampled yet, call .sample_prior_predictive() first",
    )
    prior_predictive = create_idata_accessor(
        "prior_predictive",
        "The model hasn't been sampled yet, call .sample_prior_predictive() first",
    )
    posterior = create_idata_accessor(
        "posterior", "The model hasn't been fit yet, call .fit() first"
    )

    posterior_predictive = create_idata_accessor(
        "posterior_predictive",
        "The model hasn't been fit yet, call .sample_posterior_predictive() first",
    )
    predictions = create_idata_accessor(
        "predictions",
        "Call the 'sample_posterior_predictive' method with predictions=True first.",
    )


class RegressionModelBuilder(ModelBuilder):
    """ModelBuilder class providing an easy-to-use API similar to scikit-learn for regression models.

    Training data is provided in the fit method and must follow the following convention:
    - X: Matrix containing predictor variables
    - y: Target variable array
    """

    def _validate_data(self, X, y=None):
        if y is not None:
            return check_X_y(
                X, y, accept_sparse=False, y_numeric=True, multi_output=False
            )
        else:
            return check_array(X, accept_sparse=False)

    @abstractmethod
    def _data_setter(
        self,
        X: np.ndarray | pd.DataFrame | xr.Dataset | xr.DataArray,
        y: np.ndarray | pd.Series | xr.DataArray | None = None,
    ) -> None:
        """Set new data in the model.

        Parameters
        ----------
        X : array, shape (n_obs, n_features)
            The training input samples.
        y : array, shape (n_obs,)
            The target values (real numbers).

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            def _data_setter(self, data: pd.DataFrame):
                with self.model:
                    pm.set_data({"x": X["x"].values})
                    try:  # if y values in new data
                        pm.set_data({"y_data": y.values})
                    except:  # dummies otherwise
                        pm.set_data({"y_data": np.zeros(len(data))})

        """

    @property
    @abstractmethod
    def output_var(self) -> str:
        """Returns the name of the output variable of the model.

        Returns
        -------
        output_var : str
            Name of the output variable of the model.

        """

    @abstractmethod
    def build_model(  # type: ignore[override]
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: pd.Series | np.ndarray | xr.DataArray,
        **kwargs,
    ) -> None:
        """Create an instance of `pm.Model` based on provided data and model_config.

        It attaches the model to self.model.

        Parameters
        ----------
        X : pd.DataFrame | xr.Dataset | xr.DataArray
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series | np.ndarray | xr.DataArray
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.

        See Also
        --------
        default_model_config : returns default model config

        Returns
        -------
        None

        """

    def build_from_idata(self, idata: xr.DataTree) -> None:
        """Build model from the DataTree object.

        This is part of the :func:`load` method. See :func:`load` for more larger context.

        Usually a wrapper around the :func:`build_model` method unless the model
        has some additional steps to be built.

        Parameters
        ----------
        idata : xr.DataTree
            The DataTree object to build the model from.

        """
        self.idata = idata
        dataset = idata.fit_data.dataset.to_dataframe()  # type: ignore
        X = dataset.drop(columns=[self.output_var])
        y = dataset[self.output_var]

        self.build_model(X, y)  # type: ignore

    def create_fit_data(
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: np.ndarray | pd.Series | xr.DataArray,
    ) -> xr.Dataset:
        """Create the fit_data group based on the input data."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index, name=self.output_var)

        y.name = self.output_var

        if isinstance(X, pd.DataFrame):
            X = X.to_xarray()

        if isinstance(y, pd.Series):
            y = y.to_xarray()

        return xr.merge([X, y])

    def _validate_fit_inputs(
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: pd.Series | xr.DataArray | np.ndarray | None = None,
    ) -> tuple[Any, Any]:
        """Check the X/y pair and fill in a placeholder target when none is given."""
        if (
            isinstance(y, pd.Series)
            and isinstance(X, pd.DataFrame)
            and not X.index.equals(y.index)
        ):
            raise ValueError("Index of X and y must match.")

        if y is None:
            y = np.zeros(X.shape[0])

        if self.output_var in X:
            raise ValueError(
                f"X includes a column named '{self.output_var}', which conflicts with the target variable."
            )

        return X, y

    def _prepare_fit(self, data: Any = None) -> None:
        """No-op: :meth:`fit` builds the model from the X/y pair before delegating."""

    def create_fit_data_group(self) -> xr.Dataset | None:
        """Return the ``fit_data`` group built by :meth:`fit` from the X/y pair."""
        return getattr(self, "_fit_data_group", None)

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: pd.Series | xr.DataArray | np.ndarray | None = None,
        *,
        method: SamplingMethod = "mcmc",
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> xr.DataTree:
        """Fit a model using the data passed as a parameter.

        Thin wrapper around :meth:`ModelFitter.fit` supporting the X/y data convention.
        Sets attrs to inference data of the model.

        Parameters
        ----------
        X : array-like | array, shape (n_obs, n_features)
            The training input samples. If scikit-learn is available, array-like, otherwise array.
        y : array-like | array, shape (n_obs,)
            The target values (real numbers). If scikit-learn is available, array-like, otherwise array.
        method : str
            Method used to fit the model. One of ``"mcmc"``, ``"map"``, ``"demz"``,
            ``"advi"`` or ``"fullrank_advi"``. See :meth:`ModelFitter.fit`.
        progressbar : bool, optional
            Specifies whether the fit progress bar should be displayed. Defaults to True.
        random_seed : Optional[RandomState]
            Provides sampler with initial random seed for obtaining reproducible samples.
        sample_kwargs : dict, optional
            Only used by the variational methods; forwarded to ``Approximation.sample``.
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : xr.DataTree
            Returns inference data of the fitted model.

        Examples
        --------
        .. code-block:: python

            model = MyModel()
            idata = model.fit(X, y)
            Auto-assigning NUTS sampler...
            Initializing NUTS using jitter+adapt_diag...

        """
        X, y = self._validate_fit_inputs(X, y)

        if not hasattr(self, "model"):
            self.build_model(X, y)

        self._fit_data_group = self.create_fit_data(X, y)

        return super().fit(
            method=method,
            progressbar=progressbar,
            random_seed=random_seed,
            sample_kwargs=sample_kwargs,
            **kwargs,
        )

    def predict(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
        extend_idata: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Use a model to predict on unseen data and return point prediction of all the samples.

        The point prediction for each input row is the expected output value, computed as the mean of MCMC samples.

        Parameters
        ----------
        X : array-like | array, shape (n_pred, n_features)
            The input data used for prediction. If scikit-learn is available, array-like, otherwise array.
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        **kwargs: Additional arguments to pass to sample_posterior_predictive method

        Returns
        -------
        ndarray, shape (n_pred,)
            Predicted output corresponding to input X.

        Examples
        --------
        .. code-block:: python

            model = MyModel()
            idata = model.fit(X, y)
            x_pred = []
            prediction_data = pd.DataFrame({"input": x_pred})
            pred_mean = model.predict(prediction_data)

        """
        posterior_predictive_samples = self.sample_posterior_predictive(
            X,
            extend_idata=extend_idata,
            combined=False,
            **kwargs,
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )

        posterior_means = posterior_predictive_samples[self.output_var].mean(
            dim=["chain", "draw"], keep_attrs=True
        )
        return posterior_means.data

    def approximate_fit(
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: pd.Series | xr.DataArray | np.ndarray | None = None,
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        *,
        fit_kwargs: dict[str, Any] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataTree:
        """Fit a model using Variational Inference and return a DataTree.

        This performs variational inference via `pymc.fit`, then draws posterior samples
        from the fitted approximation via `Approximation.sample`, returning an
        `xr.DataTree` compatible with the rest of the API (same structure as `.fit`).

        Parameters
        ----------
        X : array-like | array, shape (n_obs, n_features)
            The training input samples. If scikit-learn is available, array-like, otherwise array.
        y : array-like | array, shape (n_obs,)
            The target values (real numbers). If scikit-learn is available, array-like, otherwise array.
        progressbar : bool, optional
            Specifies whether the fitting/sample progress bar should be displayed. Defaults to True.
        random_seed : Optional[RandomState]
            Provides stochastic procedures with initial random seed for reproducibility.
        fit_kwargs : dict, optional
            Extra keyword arguments forwarded to `pymc.fit` (e.g., {"n": 10_000, "method": "advi"}).
        sample_kwargs : dict, optional
            Extra keyword arguments forwarded to `Approximation.sample` (e.g., {"draws": 1_000}).

        Returns
        -------
        xr.DataTree
            DataTree of the variationally fitted model.

        .. deprecated:: 1.0.0
            Use ``fit(X, y, method="advi")`` instead.
        """
        warnings.warn(
            "`approximate_fit` is deprecated and will be removed in a future release. "
            'Use `fit(X, y, method="advi")` instead.',
            FutureWarning,
            stacklevel=2,
        )
        _fit_kwargs = dict(fit_kwargs or {})
        method = _fit_kwargs.pop("method", "advi")

        return self.fit(
            X,
            y,
            progressbar=progressbar,
            random_seed=random_seed,
            method=method,
            sample_kwargs=sample_kwargs,
            **_fit_kwargs,
        )

    def sample_prior_predictive(
        self,
        X,
        y=None,
        samples: int | None = None,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs,
    ):
        """Sample from the model's prior predictive distribution.

        Parameters
        ----------
        X : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution.
        y : array, shape (n_pred,), optional
            The target values (real numbers) used for prediction using prior distribution.
            If not set, defaults to an array of zeros.
        samples : int
            Number of samples from the prior parameter distributions to generate.
            If not set, uses sampler_config['draws'] if that is available, otherwise defaults to 500.
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Boolean
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to pymc.sample_prior_predictive

        Returns
        -------
        prior_predictive_samples : DataArray, shape (n_pred, samples)
            Prior predictive samples for each input X

        """
        if y is None:
            y = np.zeros(len(X))
        if samples is None:
            samples = self.sampler_config.get("draws", 500)

        if not hasattr(self, "model"):
            self.build_model(X, y)

        with self.model:  # sample with new input data
            prior_pred: xr.DataTree = pm.sample_prior_predictive(
                draws=samples, **kwargs
            )
            prior_pred["/prior"].attrs["pymc_marketing_version"] = __version__
            prior_pred["/prior_predictive"].attrs["pymc_marketing_version"] = (
                __version__
            )
            self.set_idata_attrs(prior_pred)

        if extend_idata:
            if self.idata is not None:
                self.idata.update(prior_pred)
            else:
                self.idata = prior_pred

        result = az.extract(
            prior_pred, "prior_predictive", combined=combined, keep_dataset=True
        )
        return result

    def sample_posterior_predictive(
        self,
        X,
        extend_idata: bool = True,
        combined: bool = True,
        **sample_posterior_predictive_kwargs,
    ):
        """Sample from the model's posterior predictive distribution.

        Parameters
        ----------
        X : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution..
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Boolean
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **sample_posterior_predictive_kwargs: Additional arguments to pass to pymc.sample_posterior_predictive

        Returns
        -------
        posterior_predictive_samples : DataArray, shape (n_pred, samples)
            Posterior predictive samples for each input X

        """
        self._data_setter(X)

        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )

        if extend_idata:
            self.idata.update(post_pred)  # type: ignore

        variable_name = (
            "predictions"
            if sample_posterior_predictive_kwargs.get("predictions")
            else "posterior_predictive"
        )

        result = az.extract(
            post_pred, variable_name, combined=combined, keep_dataset=True
        )
        return result

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
        extend_idata: bool = True,
        combined: bool = False,
        **kwargs,
    ) -> xr.DataArray:
        """Alias for `predict_posterior`, for consistency with scikit-learn probabilistic estimators."""
        return self.predict_posterior(X, extend_idata, combined, **kwargs)

    def predict_posterior(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """Generate posterior predictive samples on unseen data.

        Parameters
        ----------
        X : array-like | array, shape (n_pred, n_features)
            The input data used for prediction. If scikit-learn is available, array-like, otherwise array.
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Boolean
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to sample_posterior_predictive method

        Returns
        -------
        y_pred : DataArray
            Posterior predictive samples for each input X.
            Shape is (n_pred, chains * draws) if combined is True, otherwise (chains, draws, n_pred).

        """
        X = self._validate_data(X)
        posterior_predictive_samples = self.sample_posterior_predictive(
            X, extend_idata, combined, **kwargs
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )

        return posterior_predictive_samples[self.output_var]
