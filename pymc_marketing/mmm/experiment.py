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
"""CausalPy experiment bridge for MMM lift test calibration.

This module provides an integration layer between CausalPy causal inference
experiments and the MMM lift test calibration workflow. It allows users to
run quasi-experiments (e.g. Interrupted Time Series, Synthetic Control,
Difference-in-Differences) and convert results into the DataFrame format
expected by :meth:`MMM.add_lift_test_measurements`.

Examples
--------
Run a Synthetic Control experiment and convert to a lift test:

.. code-block:: python

    from pymc_marketing.mmm.experiment import run_experiment

    result = run_experiment(
        experiment_type="sc",
        data=df,
        treatment_time=70,
        formula="actual ~ 0 + a + b + c + d + e + f + g",
    )

    df_lift = result.to_lift_test(
        channel="tv",
        x=1000.0,
        delta_x=200.0,
        geo="US",
    )

    mmm.add_lift_test_measurements(df_lift)

"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import causalpy


_CAUSALPY_IMPORT_ERROR_MSG = (
    "CausalPy is required for running experiments. "
    "Install it with: pip install causalpy>=0.7.0 "
    "or pip install pymc-marketing[experiment]"
)


def _import_causalpy() -> Any:
    """Lazily import CausalPy with a helpful error message.

    Returns
    -------
    module
        The ``causalpy`` module.

    Raises
    ------
    ImportError
        If ``causalpy`` is not installed.
    """
    try:
        import causalpy

        return causalpy
    except ImportError:
        raise ImportError(_CAUSALPY_IMPORT_ERROR_MSG) from None


class ExperimentType(StrEnum):
    """Supported CausalPy experiment types.

    Maps short aliases to CausalPy experiment class names.

    Attributes
    ----------
    ITS : str
        Interrupted Time Series.
    SC : str
        Synthetic Control.
    DID : str
        Difference-in-Differences.
    RD : str
        Regression Discontinuity.
    """

    ITS = "its"
    SC = "sc"
    DID = "did"
    RD = "rd"

    def get_experiment_class(self) -> type:
        """Return the CausalPy experiment class for this type.

        Returns
        -------
        type
            A CausalPy experiment class.

        Raises
        ------
        ImportError
            If ``causalpy`` is not installed.
        """
        cp = _import_causalpy()
        mapping: dict[ExperimentType, type] = {
            ExperimentType.ITS: cp.InterruptedTimeSeries,
            ExperimentType.SC: cp.SyntheticControl,
            ExperimentType.DID: cp.DifferenceInDifferences,
            ExperimentType.RD: cp.RegressionDiscontinuity,
        }
        return mapping[self]


class ExperimentResult:
    """Wrapper around a CausalPy experiment result.

    Provides convenience methods to access the causal effect estimates
    and convert them into the lift test DataFrame format expected by
    :meth:`MMM.add_lift_test_measurements`.

    Parameters
    ----------
    result : causalpy experiment instance
        The CausalPy experiment result (e.g. ``InterruptedTimeSeries``).
    experiment_type : ExperimentType
        The type of experiment that was run.

    Attributes
    ----------
    result : causalpy experiment instance
        The underlying CausalPy result object.
    experiment_type : ExperimentType
        The experiment type.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm.experiment import ExperimentResult, ExperimentType

        experiment_result = ExperimentResult(
            result=causalpy_result,
            experiment_type=ExperimentType.ITS,
        )
        experiment_result.summary()
        df_lift = experiment_result.to_lift_test(channel="tv", x=1000.0, delta_x=200.0)

    """

    def __init__(
        self,
        result: causalpy.InterruptedTimeSeries
        | causalpy.SyntheticControl
        | causalpy.DifferenceInDifferences
        | causalpy.RegressionDiscontinuity,
        experiment_type: ExperimentType,
    ) -> None:
        self.result = result
        self.experiment_type = experiment_type

    def summary(self) -> None:
        """Print a summary of the experiment results.

        Delegates to the underlying CausalPy result's ``summary`` method.
        """
        self.result.summary()

    def effect_summary(self, **kwargs: Any) -> Any:
        """Get a structured effect summary.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to ``result.effect_summary()``.
            Common arguments include ``direction`` (``"increase"``,
            ``"decrease"``, ``"two-sided"``), ``alpha`` (HDI level),
            and ``min_effect`` (ROPE threshold).

        Returns
        -------
        object
            Effect summary with ``.table`` (DataFrame) and ``.text``
            (string) attributes.
        """
        return self.result.effect_summary(**kwargs)

    def plot(self, **kwargs: Any) -> tuple:
        """Plot the experiment results.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to ``result.plot()``.

        Returns
        -------
        tuple
            A ``(fig, ax)`` tuple from matplotlib.
        """
        return self.result.plot(**kwargs)

    @property
    def idata(self) -> Any:
        """Access the ArviZ InferenceData object.

        Returns
        -------
        arviz.InferenceData
            The inference data from the fitted experiment model.
        """
        return self.result.idata

    def _get_causal_impact(self) -> tuple[float, float]:
        """Extract the causal impact mean and standard deviation.

        Returns
        -------
        tuple[float, float]
            A ``(mean, std)`` tuple of the posterior causal impact.

        Raises
        ------
        ValueError
            If the causal impact cannot be extracted for the experiment type.
        """
        if self.experiment_type in (ExperimentType.ITS, ExperimentType.SC):
            post_impact = self.result.post_impact
            # Sum over time to get total lift
            total_impact = post_impact.sum("obs_ind")
            mean = float(total_impact.mean().values)
            std = float(total_impact.std().values)
            return mean, std
        elif self.experiment_type == ExperimentType.DID:
            causal_impact = self.result.causal_impact
            mean = float(causal_impact.mean().values)
            std = float(causal_impact.std().values)
            return mean, std
        elif self.experiment_type == ExperimentType.RD:
            effect = self.result.effect_summary()
            table = effect.table
            mean = float(table["mean"].iloc[0])
            # Use the HDI width as a proxy for std
            hdi_lower = float(table["hdi_lower"].iloc[0])
            hdi_upper = float(table["hdi_upper"].iloc[0])
            # Approximate std from 94% HDI (default): HDI width ~ 2 * 1.88 * std
            std = (hdi_upper - hdi_lower) / (2 * 1.88)
            return mean, std
        else:
            raise ValueError(
                f"Cannot extract causal impact for experiment type: "
                f"{self.experiment_type}"
            )

    def to_lift_test(
        self,
        channel: str,
        x: float,
        delta_x: float,
        **dim_kwargs: str,
    ) -> pd.DataFrame:
        """Convert experiment results to a lift test DataFrame.

        Produces a single-row DataFrame in the format expected by
        :meth:`MMM.add_lift_test_measurements`, with columns
        ``channel``, ``x``, ``delta_x``, ``delta_y``, ``sigma``,
        plus any additional dimension columns.

        Parameters
        ----------
        channel : str
            The marketing channel name. Must match one of the MMM's
            ``channel_columns``.
        x : float
            The baseline spend level for the channel during the
            experiment period.
        delta_x : float
            The change in spend during the experiment.
        **dim_kwargs : str
            Additional dimension values, e.g. ``geo="US"``. Keys must
            match the MMM's ``dims``.

        Returns
        -------
        pd.DataFrame
            A single-row DataFrame with columns: ``channel``, ``x``,
            ``delta_x``, ``delta_y``, ``sigma``, and any dimension
            columns from ``dim_kwargs``.

        Examples
        --------
        .. code-block:: python

            df_lift = result.to_lift_test(
                channel="tv",
                x=1000.0,
                delta_x=200.0,
                geo="US",
            )

        """
        mean, std = self._get_causal_impact()

        data: dict[str, Any] = {
            "channel": [channel],
            "x": [x],
            "delta_x": [delta_x],
            "delta_y": [mean],
            "sigma": [std],
        }

        for dim_name, dim_value in dim_kwargs.items():
            data[dim_name] = [dim_value]

        return pd.DataFrame(data)


def run_experiment(
    experiment_type: str | ExperimentType,
    data: pd.DataFrame,
    **kwargs: Any,
) -> ExperimentResult:
    """Run a CausalPy experiment and return a wrapped result.

    This is the main entry point for running causal experiments that
    can be used to calibrate an MMM via lift tests.

    Parameters
    ----------
    experiment_type : str or ExperimentType
        The type of experiment to run. Accepts string aliases
        (``"its"``, ``"sc"``, ``"did"``, ``"rd"``) or
        :class:`ExperimentType` enum values.
    data : pd.DataFrame
        The experiment data to pass to CausalPy.
    **kwargs
        Additional keyword arguments passed directly to the CausalPy
        experiment constructor. Common arguments include:

        - ``treatment_time``: When the treatment/intervention started.
        - ``formula``: Patsy formula for the model.
        - ``model``: A CausalPy model instance
          (e.g. ``cp.pymc_models.LinearRegression()``).
        - For Synthetic Control: ``control_units``, ``treated_units``.
        - For DiD: ``time_variable_name``, ``group_variable_name``.
        - For RD: ``treatment_threshold``, ``running_variable_name``.

    Returns
    -------
    ExperimentResult
        A wrapped result with methods for summarizing, plotting, and
        converting to lift test format.

    Raises
    ------
    ImportError
        If ``causalpy`` is not installed.
    ValueError
        If ``experiment_type`` is not a valid type.

    Examples
    --------
    Run an Interrupted Time Series experiment:

    .. code-block:: python

        import causalpy as cp
        from pymc_marketing.mmm.experiment import run_experiment

        result = run_experiment(
            experiment_type="its",
            data=df,
            treatment_time=pd.Timestamp("2024-01-01"),
            formula="y ~ 1 + t",
            model=cp.pymc_models.LinearRegression(),
        )

        result.summary()
        fig, ax = result.plot()

    Run a Synthetic Control experiment:

    .. code-block:: python

        result = run_experiment(
            experiment_type="sc",
            data=df,
            treatment_time=70,
            formula="actual ~ 0 + a + b + c",
            model=cp.pymc_models.WeightedSumFitter(),
        )

        df_lift = result.to_lift_test(channel="tv", x=1000.0, delta_x=200.0)

    """
    # Ensure CausalPy is available
    _import_causalpy()

    if isinstance(experiment_type, str):
        try:
            exp_type = ExperimentType(experiment_type.lower())
        except ValueError:
            valid = [e.value for e in ExperimentType]
            raise ValueError(
                f"Invalid experiment type: {experiment_type!r}. "
                f"Valid types are: {valid}"
            ) from None
    else:
        exp_type = experiment_type

    experiment_class = exp_type.get_experiment_class()
    result = experiment_class(data=data, **kwargs)

    return ExperimentResult(result=result, experiment_type=exp_type)
