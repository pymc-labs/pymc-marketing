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
"""Names of the MMM: dimensions, variables, ``idata.attrs`` keys and groups.

These are the names that more than one place in :mod:`pymc_marketing.mmm` or
:mod:`pymc_marketing.data.idata` has to agree on. Each is a ``Final`` string
attribute, so ``mypy`` reports a typo as ``attr-defined`` and the runtime value
stays a plain ``str``. The values are public API: users type them and saved
models store them, so they must not change.

The module imports nothing from ``pymc_marketing`` except
:mod:`pymc_marketing.names`, so :mod:`pymc_marketing.data.idata` and
:mod:`pymc_marketing.mmm` can both import it without a cycle.
"""

from typing import Final

from pymc_marketing import names


class Dim:
    """Dimension names of the MMM and of the results it stores.

    ``DATE``, ``CHANNEL``, ``CONTROL`` and ``FOURIER_MODE`` are the dims of the
    model (see :data:`RESERVED_DIMS`). ``TIME_SINCE_EXPOSURE`` is the lag dim of a
    sampled adstock curve, ``CV`` the fold dim of the cross-validation results and
    ``SWEEP`` the dim of a sensitivity sweep.
    """

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
    """Variable names of the MMM that more than one place reads or writes.

    Most are variables of the PyMC model. ``CHANNEL_SPEND`` exists only in
    ``idata.constant_data``, ``ALLOCATION`` and ``TOTAL_ALLOCATION`` are variables
    of the dataset that ``MMM.sample_response_distribution`` returns, and
    ``METADATA`` is the variable of the ``cv_metadata`` group.
    """

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
    METADATA: Final = "metadata"


class DatasetVar:
    """Data variables of ``MMM.xarray_dataset``, which also key ``MMM.scalers``.

    :func:`~pymc_marketing.mmm.data_conversion.to_mmm_dataset` names the input
    data with them. They are not the model's data variables: the model reads
    ``DatasetVar.CHANNEL`` into ``Var.CHANNEL_DATA``.
    """

    CHANNEL: Final = "_channel"
    TARGET: Final = "_target"
    CONTROL: Final = "_control"


class Attr(names.Attr):
    """``idata.attrs`` keys of the MMM that more than one place reads or writes.

    The shared keys are inherited. A copy of an ``MMM.__init__`` parameter that
    only ``MMM.create_idata_attrs`` and ``MMM.attrs_to_init_kwargs`` use stays a
    literal in those two methods. ``ADSTOCK``, ``SATURATION``,
    ``TIME_VARYING_INTERCEPT`` and ``TIME_VARYING_MEDIA`` are such copies that
    :mod:`pymc_marketing.serialization_migration` also reads.
    """

    LINK: Final = "link"
    COST_PER_UNIT: Final = "cost_per_unit"
    MU_EFFECTS: Final = "mu_effects"
    ORIGINAL_SCALE_VARS: Final = "original_scale_vars"
    TIME_AGGREGATION: Final = "time_aggregation"
    LIFT_TEST_CALIBRATIONS: Final = "lift_test_calibrations"
    COST_PER_TARGET_CALIBRATIONS: Final = "cost_per_target_calibrations"
    ADSTOCK: Final = "adstock"
    SATURATION: Final = "saturation"
    TIME_VARYING_INTERCEPT: Final = "time_varying_intercept"
    TIME_VARYING_MEDIA: Final = "time_varying_media"


class Group(names.Group):
    """InferenceData groups the MMM adds, plus the shared ones."""

    SENSITIVITY_ANALYSIS: Final = "sensitivity_analysis"
    CV_METADATA: Final = "cv_metadata"


ORIGINAL_SCALE_SUFFIX: Final = "_original_scale"
"""Suffix of the original-scale copy of a variable, see :func:`original_scale_name`."""


def original_scale_name(name: str) -> str:
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

        from pymc_marketing.names.mmm import Var, original_scale_name

        original_scale_name(Var.CHANNEL_CONTRIBUTION)
        # 'channel_contribution_original_scale'
    """
    return f"{name}{ORIGINAL_SCALE_SUFFIX}"


EFFECT_CONTRIBUTION_SUFFIX: Final = "_effect_contribution"
"""Suffix of the default contribution name of an additive effect, see :func:`effect_contribution_name`."""


def effect_contribution_name(prefix: str) -> str:
    """Return the default name of the contribution variable of an additive effect.

    ``MuEffect.contribution_var_name`` returns this name by default, and the
    effects that keep the default register their contribution under it.
    ``FourierEffect`` and ``EventAdditiveEffect`` register names of their own.

    Parameters
    ----------
    prefix : str
        Prefix of the effect.

    Returns
    -------
    str
        ``prefix`` followed by :data:`EFFECT_CONTRIBUTION_SUFFIX`.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.names.mmm import effect_contribution_name

        effect_contribution_name("trend")
        # 'trend_effect_contribution'
    """
    return f"{prefix}{EFFECT_CONTRIBUTION_SUFFIX}"
