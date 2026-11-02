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
"""Names of the MMM: dimensions, model variables, ``idata.attrs`` keys and groups.

These are the names one module of :mod:`pymc_marketing.mmm` creates and another
module (or :mod:`pymc_marketing.data.idata`) reads back. Each is a member of a
:class:`~pymc_marketing.names.Name` enum, so ``mypy`` reports a typo as
``attr-defined``; see :class:`~pymc_marketing.names.Name` for how members behave
when saved and loaded. The values are public API: users type them and saved
models store them, so they must not change.

The module lives in :mod:`pymc_marketing.data.idata` rather than in
:mod:`pymc_marketing.mmm` because :mod:`pymc_marketing.data.idata` is imported
before :mod:`pymc_marketing.mmm` and cannot import it without a cycle. It imports
nothing from ``pymc_marketing`` except :mod:`pymc_marketing.names`.
"""

from enum import unique
from typing import Final

from pymc_marketing import names
from pymc_marketing.names import Name


@unique
class Dim(Name):
    """Dimension names of the MMM."""

    DATE = "date"
    CHANNEL = "channel"
    CONTROL = "control"
    FOURIER_MODE = "fourier_mode"
    TIME_SINCE_EXPOSURE = "time since exposure"
    CV = "cv"
    SWEEP = "sweep"


RESERVED_DIMS: Final = frozenset({Dim.DATE, Dim.CHANNEL, Dim.CONTROL, Dim.FOURIER_MODE})
"""Dimensions the MMM creates itself, so a user-supplied dim cannot take their names."""


@unique
class Var(Name):
    """Model variable names that other modules read back."""

    CHANNEL_DATA = "channel_data"
    TARGET_DATA = "target_data"
    CONTROL_DATA = "control_data"
    CHANNEL_SCALE = "channel_scale"
    TARGET_SCALE = "target_scale"
    CHANNEL_SPEND = "channel_spend"
    TIME_INDEX = "time_index"
    DAYOFYEAR = "dayofyear"
    MU = "mu"
    MU_ORIGINAL_SCALE = "mu_original_scale"
    Y = "y"
    Y_ORIGINAL_SCALE = "y_original_scale"
    CHANNEL_CONTRIBUTION = "channel_contribution"
    CHANNEL_CONTRIBUTION_ORIGINAL_SCALE = "channel_contribution_original_scale"
    CHANNEL_INCREMENTAL_CONTRIBUTION = "channel_incremental_contribution"
    CONTROL_CONTRIBUTION = "control_contribution"
    CONTROL_CONTRIBUTION_ORIGINAL_SCALE = "control_contribution_original_scale"
    INTERCEPT_BASELINE = "intercept_baseline"
    INTERCEPT_CONTRIBUTION = "intercept_contribution"
    INTERCEPT_CONTRIBUTION_ORIGINAL_SCALE = "intercept_contribution_original_scale"
    YEARLY_SEASONALITY_CONTRIBUTION = "yearly_seasonality_contribution"
    YEARLY_SEASONALITY_CONTRIBUTION_ORIGINAL_SCALE = (
        "yearly_seasonality_contribution_original_scale"
    )
    TOTAL_MEDIA_CONTRIBUTION_ORIGINAL_SCALE = "total_media_contribution_original_scale"
    TOTAL_RESPONSE_ORIGINAL_SCALE = "total_response_original_scale"
    ALLOCATION = "allocation"
    TOTAL_ALLOCATION = "total_allocation"


@unique
class Attr(Name):
    """``idata.attrs`` keys the MMM writes and reads back, plus the shared ones.

    An enum cannot extend :class:`pymc_marketing.names.Attr`, so the shared keys
    are members here too, with their value taken from there. They compare equal
    to the root members but are separate members: a key added to
    :class:`pymc_marketing.names.Attr` must be added here too.
    """

    ID = names.Attr.ID.value
    MODEL_TYPE = names.Attr.MODEL_TYPE.value
    VERSION = names.Attr.VERSION.value
    SAMPLER_CONFIG = names.Attr.SAMPLER_CONFIG.value
    MODEL_CONFIG = names.Attr.MODEL_CONFIG.value
    SERIALIZATION_VERSION = names.Attr.SERIALIZATION_VERSION.value
    LINK = "link"
    COST_PER_UNIT = "cost_per_unit"
    MU_EFFECTS = "mu_effects"
    ORIGINAL_SCALE_VARS = "original_scale_vars"
    TIME_AGGREGATION = "time_aggregation"
    LIFT_TEST_CALIBRATIONS = "lift_test_calibrations"
    COST_PER_TARGET_CALIBRATIONS = "cost_per_target_calibrations"


@unique
class Group(Name):
    """InferenceData groups the MMM adds, plus the shared ones.

    An enum cannot extend :class:`pymc_marketing.names.Group`, so the shared groups
    are members here too, with their value taken from there. They compare equal
    to the root members but are separate members: a group added to
    :class:`pymc_marketing.names.Group` must be added here too.
    """

    FIT_DATA = names.Group.FIT_DATA.value
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    CV_METADATA = "cv_metadata"


ORIGINAL_SCALE_SUFFIX: Final = "_original_scale"
"""Suffix of the original-scale copy of a variable, see :func:`original_scale`."""


def original_scale(name: str) -> str:
    """Return the name of the original-scale version of a variable.

    Parameters
    ----------
    name : str
        Name of the variable on the scaled target.

    Returns
    -------
    str
        ``name`` followed by :data:`ORIGINAL_SCALE_SUFFIX`.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.data.idata.names import Var, original_scale

        original_scale(Var.CHANNEL_CONTRIBUTION)
        # 'channel_contribution_original_scale'
    """
    return f"{name}{ORIGINAL_SCALE_SUFFIX}"
