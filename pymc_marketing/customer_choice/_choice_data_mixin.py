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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

import pandas as pd
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

    if TYPE_CHECKING:
        # Provided by the model-builder base that these classes are mixed with.
        def build_model(self, **kwargs: Any) -> None: ...

        @classmethod
        def _model_config_formatting(cls, model_config: dict) -> dict: ...

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

        Returns
        -------
        xr.Dataset
            Choice data as an xarray Dataset with an ``obs`` dimension.
        """
        choice_df = self._require_choice_df()
        return choice_df.rename_axis("obs").to_xarray()

    def build_from_idata(self, idata: xr.DataTree) -> None:
        """Build the model from a loaded DataTree.

        Parameters
        ----------
        idata : xr.DataTree
            Loaded inference data.
        """
        self.choice_df = idata["fit_data"].dataset.to_dataframe().rename_axis(None)
        if not hasattr(self, "model"):
            self.build_model()
