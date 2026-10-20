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
"""Shared choice-data handling for the discrete-choice logit models."""

import json
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr


class ChoiceDataMixin:
    """Choice-data and settings lifecycle shared by the logit models.

    This mixin keeps the pieces that :class:`MNLogit` and :class:`NestedLogit`
    must not drift apart: validation of the required constructor settings, the
    "choice data is required" guard, the serialisation of constructor
    arguments to and from the ``DataTree`` attributes, and the ``fit_data``
    group used to restore the choice data on load.

    It deliberately does not define ``__init__``. Each class keeps its own
    constructor signature, so the ``super().__init__(...)`` chain still
    reaches the model-builder base.

    Attributes
    ----------
    choice_df : pd.DataFrame or None
        The choice data set at construction or :meth:`fit` time.
    utility_equations : list of str
        The utility equations set at construction or :meth:`fit` time.
    """

    choice_df: pd.DataFrame | None
    utility_equations: list[str]

    #: Name of the model's fixed-covariate data container.
    #: It is ``"F"`` on :class:`MNLogit` and ``"W"`` on :class:`NestedLogit`.
    _fixed_data_name: str

    #: Name of the attribute holding the model's design matrix.
    #: It is ``"X"`` on :class:`MNLogit` and ``"X_data"`` on :class:`NestedLogit`.
    _X_data_name: str

    if TYPE_CHECKING:
        # Provided by the model-builder base that these classes are mixed with.
        model: pm.Model
        coords: dict[str, Any]
        F: np.ndarray | None
        y: np.ndarray

        def build_model(self, **kwargs: Any) -> None: ...

        def preprocess_model_data(
            self, choice_df: pd.DataFrame, utility_equations: list[str]
        ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]: ...

        @classmethod
        def _model_config_formatting(cls, model_config: dict) -> dict: ...

    def _set_model_data(
        self, X: np.ndarray, F: np.ndarray | None, y: np.ndarray
    ) -> None:
        """Point the fitted model's data containers at preprocessed arrays.

        The ``obs`` coordinates are refreshed whenever the arrays hold a
        different number of rows than the model currently declares, which is
        what lets a prediction run on a data set with a changed row count.

        Parameters
        ----------
        X : np.ndarray
            Alternative-specific covariates, shape ``(n_obs, n_alts, n_cov)``.
        F : np.ndarray or None
            Fixed covariates, or ``None`` when the model has none.
        y : np.ndarray
            Encoded observed choices, shape ``(n_obs,)``.
        """
        data: dict[str, Any] = {"X": X, "y": y}
        if F is not None and len(F) > 0:
            data[self._fixed_data_name] = F

        n_obs = len(X)
        coords = None
        if len(self.model.coords["obs"]) != n_obs:
            coords = {"obs": range(n_obs)}

        with self.model:
            pm.set_data(data, coords=coords)

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Return the design matrix, fixed covariates and outcomes stored at build time.

        Returns
        -------
        tuple
            The arrays ``build_model`` stored from preprocessing, in the order
            ``(X, F, y)``. ``F`` is ``None`` when the model has no fixed
            covariates.
        """
        return getattr(self, self._X_data_name), self.F, self.y

    @contextmanager
    def _prediction_model_data(self, choice_df: pd.DataFrame | None) -> Iterator[None]:
        """Point the model at prediction data, resetting it to the training data on exit.

        The ``None`` case leaves the model untouched. Otherwise the new data is
        preprocessed and set on the model, and the training arrays stored by
        :meth:`build_model` are reset on exit without re-preprocessing, even
        when the body fails. If the reset fails too, the original error is re-raised
        with the reset failure attached as a note, so the traceback does not
        claim the reset caused it.
        """
        if choice_df is None:
            yield
            return

        try:
            X, F, y = self.preprocess_model_data(choice_df, self.utility_equations)
            self._set_model_data(X, F, y)
            yield
        except BaseException as error:
            try:
                self._set_model_data(*self._training_arrays())
            except Exception as reset_error:
                error.add_note(f"Resetting the model data also failed: {reset_error!r}")
            raise
        else:
            self._set_model_data(*self._training_arrays())

    @classmethod
    def _raise_missing_settings(cls, **settings: Any) -> NoReturn:
        """Raise listing the passed settings that are ``None``.

        The settings are passed in the order they should appear in the error
        message.

        Raises
        ------
        ValueError
            If any of the passed settings is ``None``.
        """
        missing = [name for name, value in settings.items() if value is None]
        names = list(settings)
        provided = (
            names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        )
        raise ValueError(
            f"Missing required argument(s): {', '.join(missing)}. "
            f"Provide {provided} when constructing {cls.__name__}."
        )

    @classmethod
    def _raise_if_choice_df_not_dataframe(
        cls, choice_df: Any, settings: Sequence[str]
    ) -> None:
        """Reject a non-DataFrame value in the first constructor slot.

        The new-style call has no positional data slot, so
        ``Model(utility_equations, depvar, covariates)`` would otherwise bind
        the utility equations to ``choice_df`` and report a misleading missing
        argument further down. Only ``None`` (no data) and a
        :class:`pandas.DataFrame` (the deprecated positional data) are valid.

        Parameters
        ----------
        choice_df : Any
            The value bound to the first constructor slot.
        settings : Sequence of str
            Keyword settings to name in the error message, in the order they
            should appear.

        Raises
        ------
        TypeError
            If ``choice_df`` is neither ``None`` nor a :class:`pandas.DataFrame`.
        """
        if choice_df is None or isinstance(choice_df, pd.DataFrame):
            return
        names = [f"`{name}`" for name in settings]
        provided = (
            names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        )
        kwargs = ", ".join(f"{name}=..." for name in settings)
        raise TypeError(
            f"{cls.__name__} received a non-DataFrame value in the choice data "
            f"slot. Pass {provided} as keyword arguments, e.g. "
            f"{cls.__name__}({kwargs})."
        )

    def _warn_if_choice_df_at_init(self, choice_df: pd.DataFrame | None) -> None:
        """Warn that passing ``choice_df`` to the constructor is deprecated."""
        if choice_df is None:
            return
        warnings.warn(
            "Passing the choice data as the first argument to "
            f"{type(self).__name__} is deprecated and will be removed in a "
            "future release. Pass it to `fit(choice_df)` instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _require_choice_df(self) -> pd.DataFrame:
        """Return the stored choice data, raising if none has been provided.

        Returns
        -------
        pd.DataFrame
            The choice data set at construction or fit time.

        Raises
        ------
        ValueError
            If no choice data has been provided to the constructor or
            :meth:`fit`.
        """
        if self.choice_df is None:
            raise ValueError(
                "Choice data is required to build the model. "
                "Pass it to `fit(choice_df)` first."
            )
        return self.choice_df

    def _same_choice_data(self, choice_df: pd.DataFrame) -> bool:
        """Whether ``choice_df`` holds the same values as the stored data.

        Values are compared exactly (no tolerance) while ignoring dtype, index
        type, column type and names. A frame restored from NetCDF can change
        dtype (datetime resolution, object to string) and the index name, so
        those are ignored, but any change to the values is caught.

        Returns
        -------
        bool
            ``True`` if the stored data is not ``None`` and matches by value.
        """
        if self.choice_df is None:
            return False
        try:
            pd.testing.assert_frame_equal(
                self.choice_df,
                choice_df,
                check_dtype=False,
                check_index_type=False,
                check_column_type=False,
                check_names=False,
                check_exact=True,
            )
        except AssertionError:
            return False
        return True

    def _set_choice_inputs(
        self,
        choice_df: pd.DataFrame | None = None,
        utility_equations: list[str] | None = None,
    ) -> None:
        """Store choice data and utility equations, guarding a built model.

        ``None`` leaves the stored value untouched and runs no check. Non-``None``
        values are checked against the already-built model before either is
        stored, so a failing check never leaves a half-updated model behind. Both
        are then stored as copies, so later in-place edits to the caller's frame
        or list do not change what the fitted model was built on.

        Raises
        ------
        ValueError
            If the model was already built with different choice data or
            different utility equations.
        """
        if (
            choice_df is not None
            and hasattr(self, "model")
            and not self._same_choice_data(choice_df)
        ):
            raise ValueError(
                "The model was built with different choice data. "
                "Create a new model instance to fit new data."
            )
        if (
            utility_equations is not None
            and hasattr(self, "model")
            and utility_equations != self.utility_equations
        ):
            raise ValueError(
                "The model was built with different utility equations. "
                "Create a new model instance to use different utility equations."
            )
        if choice_df is not None:
            self.choice_df = choice_df.copy()
        if utility_equations is not None:
            self.utility_equations = list(utility_equations)

    @classmethod
    def attrs_to_init_kwargs(cls, attrs: Mapping[str, Any]) -> dict[str, Any]:
        """Rehydrate the ``__init__`` kwargs from the serialised attrs.

        The choice data itself lives in the ``fit_data`` group and is restored
        by :meth:`build_from_idata`.
        """
        return {
            "utility_equations": json.loads(attrs["utility_equations"]),
            "depvar": json.loads(attrs["depvar"]),
            "covariates": json.loads(attrs["covariates"]),
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "sampler_config": json.loads(attrs["sampler_config"]),
        }

    def create_fit_data_group(self) -> xr.Dataset:
        """Build the ``fit_data`` group storing the choice data.

        Categorical columns are stored as their integer codes, with the
        categories and ``ordered`` flag kept as variable metadata. NetCDF
        cannot represent a pandas ``CategoricalDtype`` directly, and storing
        the labels as strings would turn non-string categories into ``NaN``.

        Returns
        -------
        xr.Dataset
            Choice data as an xarray Dataset with an ``obs`` dimension.
        """
        choice_df = self._require_choice_df()
        categorical_dtypes = {
            name: dtype
            for name, dtype in choice_df.dtypes.items()
            if isinstance(dtype, pd.CategoricalDtype)
        }
        if not categorical_dtypes:
            return choice_df.rename_axis("obs").to_xarray()

        encoded_df = choice_df.copy()
        for name in categorical_dtypes:
            encoded_df[name] = choice_df[name].cat.codes

        dataset = encoded_df.rename_axis("obs").to_xarray()
        for name, dtype in categorical_dtypes.items():
            dataset[name].attrs = {
                "categories": json.dumps(dtype.categories.tolist()),
                "ordered": int(bool(dtype.ordered)),
            }
        return dataset

    def build_from_idata(self, idata: xr.DataTree) -> None:
        """Build the model from a loaded DataTree.

        Categorical columns stored as integer codes are rebuilt with
        :func:`pandas.Categorical.from_codes` from their metadata.

        Parameters
        ----------
        idata : xr.DataTree
            Loaded inference data.
        """
        dataset = idata["fit_data"].dataset
        choice_df = dataset.to_dataframe().rename_axis(None)
        for name, variable in dataset.data_vars.items():
            categories = variable.attrs.get("categories")
            if categories is None:
                continue
            choice_df[name] = pd.Categorical.from_codes(
                choice_df[name].to_numpy(),
                categories=json.loads(categories),
                ordered=bool(variable.attrs["ordered"]),
            )
        self.choice_df = choice_df
        if not hasattr(self, "model"):
            self.build_model()
