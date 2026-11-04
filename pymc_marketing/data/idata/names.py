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
module (or :mod:`pymc_marketing.data.idata`) reads back. Each is a ``Final``
string attribute, so ``mypy`` reports a typo as ``attr-defined`` and the runtime
value stays a plain ``str``. The values are public API: users type them and saved
models store them, so they must not change.

The module lives in :mod:`pymc_marketing.data.idata` rather than in
:mod:`pymc_marketing.mmm` because :mod:`pymc_marketing.data.idata` is imported
before :mod:`pymc_marketing.mmm` and cannot import it without a cycle. It imports
nothing from ``pymc_marketing`` except :mod:`pymc_marketing.names`.
"""

from typing import Final

from pymc_marketing import names


class Dim:
    """Dimension names of the MMM."""

    DATE: Final = "date"
    CHANNEL: Final = "channel"
    CONTROL: Final = "control"
    FOURIER_MODE: Final = "fourier_mode"
    TIME_SINCE_EXPOSURE: Final = "time since exposure"
    CV: Final = "cv"
    SWEEP: Final = "sweep"


RESERVED_DIMS: Final = frozenset({Dim.DATE, Dim.CHANNEL, Dim.CONTROL, Dim.FOURIER_MODE})
"""Dimensions the MMM creates itself, so a user-supplied dim cannot take their names."""


class Var:
    """Model variable names that other modules read back."""

    CHANNEL_DATA: Final = "channel_data"
    TARGET_DATA: Final = "target_data"
    CONTROL_DATA: Final = "control_data"
    CHANNEL_SCALE: Final = "channel_scale"
    TARGET_SCALE: Final = "target_scale"
    CHANNEL_SPEND: Final = "channel_spend"
    TIME_INDEX: Final = "time_index"
    DAYOFYEAR: Final = "dayofyear"
    MU: Final = "mu"
    MU_ORIGINAL_SCALE: Final = "mu_original_scale"
    Y: Final = "y"
    Y_ORIGINAL_SCALE: Final = "y_original_scale"
    CHANNEL_CONTRIBUTION: Final = "channel_contribution"
    CHANNEL_CONTRIBUTION_ORIGINAL_SCALE: Final = "channel_contribution_original_scale"
    CHANNEL_INCREMENTAL_CONTRIBUTION: Final = "channel_incremental_contribution"
    CONTROL_CONTRIBUTION: Final = "control_contribution"
    CONTROL_CONTRIBUTION_ORIGINAL_SCALE: Final = "control_contribution_original_scale"
    INTERCEPT_BASELINE: Final = "intercept_baseline"
    INTERCEPT_CONTRIBUTION: Final = "intercept_contribution"
    INTERCEPT_CONTRIBUTION_ORIGINAL_SCALE: Final = (
        "intercept_contribution_original_scale"
    )
    YEARLY_SEASONALITY_CONTRIBUTION: Final = "yearly_seasonality_contribution"
    YEARLY_SEASONALITY_CONTRIBUTION_ORIGINAL_SCALE: Final = (
        "yearly_seasonality_contribution_original_scale"
    )
    TOTAL_MEDIA_CONTRIBUTION_ORIGINAL_SCALE: Final = (
        "total_media_contribution_original_scale"
    )
    TOTAL_RESPONSE_ORIGINAL_SCALE: Final = "total_response_original_scale"
    ALLOCATION: Final = "allocation"
    TOTAL_ALLOCATION: Final = "total_allocation"


class Attr(names.Attr):
    """``idata.attrs`` keys the MMM writes and reads back, plus the shared ones."""

    LINK: Final = "link"
    COST_PER_UNIT: Final = "cost_per_unit"
    MU_EFFECTS: Final = "mu_effects"
    ORIGINAL_SCALE_VARS: Final = "original_scale_vars"
    TIME_AGGREGATION: Final = "time_aggregation"
    LIFT_TEST_CALIBRATIONS: Final = "lift_test_calibrations"
    COST_PER_TARGET_CALIBRATIONS: Final = "cost_per_target_calibrations"


class Group(names.Group):
    """InferenceData groups the MMM adds, plus the shared ones."""

    SENSITIVITY_ANALYSIS: Final = "sensitivity_analysis"
    CV_METADATA: Final = "cv_metadata"


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
