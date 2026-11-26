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
"""Data conversion for the conversion funnel model.

A funnel has ordered *stages*, for example visit, signup, trial and paid. A
*step* is the move from one stage into the next, and it is labelled by the
stage it leads into. Counts belong to stages and conversion probabilities
belong to steps, so a funnel with four stages has three steps.

:func:`to_funnel_dataset` turns stage counts into the dataset the model reads:
for every step, the number of units that ``entered`` it and the number that
``converted``.
"""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pymc_marketing.names.conversion import Col, Dim, Var

__all__ = ["to_funnel_dataset"]


def to_funnel_dataset(
    data: pd.DataFrame | xr.Dataset,
    *,
    stage_order: Sequence[Hashable] | None = None,
) -> xr.Dataset:
    """Convert funnel data to the ``xr.Dataset`` that ``FunnelModel`` reads.

    :class:`~pymc_marketing.conversion.funnel.FunnelModel` calls this function
    on the data it is given, so it accepts the same two formats.

    Parameters
    ----------
    data : pandas.DataFrame or xarray.Dataset
        Counts of a closed funnel, in which every unit starts at the first stage
        and passes the stages in order. Either of the following.

        * A long ``DataFrame`` of stage counts with the columns ``stage`` and
          ``count``. Every other column is read as a group dimension, for example
          ``variant`` or ``segment``, and there must be exactly one row per stage
          and group.
        * A ``Dataset`` with the variables ``entered`` and ``converted`` on the
          same dimensions, one of which is ``step``. The units that enter a step
          must be the ones that converted at the step before it.
    stage_order : sequence of Hashable, optional
        Stage labels from the first stage to the last. Required for a
        ``DataFrame`` unless its ``stage`` column is an ordered categorical.
        Not accepted with a ``Dataset``, whose ``step`` coordinate is already
        ordered.

    Returns
    -------
    xarray.Dataset
        ``entered`` and ``converted`` as integers with the dimensions
        ``(step, *groups)``. For stage counts, a step is labelled by the stage it
        leads into: ``entered`` is the count of the previous stage and
        ``converted`` the count of that stage. Other variables of an input
        ``Dataset`` are kept as they are.

    Raises
    ------
    ValueError
        If a column, a variable, a coordinate or a count is missing, if the stage
        order is unknown or does not match the data, if a stage and group has
        more than one row, if there are fewer than two stages, if a count is
        negative, fractional or larger than the count of the stage before it, if
        more units converted than entered a step, or if the units entering a
        step are not the ones that converted at the step before it.
    TypeError
        If ``data`` is neither a ``DataFrame`` nor a ``Dataset``.

    Notes
    -----
    The checks cannot tell whether the cohort has had time to convert. Counts
    taken shortly after the last units entered undercount the late stages, and
    they still decrease from stage to stage. Use cohorts that are old enough for
    the slowest step to have happened.

    Examples
    --------
    Stage counts of an A/B test:

    .. code-block:: python

        import pandas as pd

        from pymc_marketing.conversion import to_funnel_dataset

        stages = ["visit", "signup", "paid"]
        df = pd.DataFrame(
            {
                "stage": stages * 2,
                "variant": ["control"] * 3 + ["treatment"] * 3,
                "count": [1000, 300, 60, 1000, 330, 75],
            }
        )
        to_funnel_dataset(df, stage_order=stages)

    The same data as explicit pairs:

    .. code-block:: python

        import xarray as xr

        ds = xr.Dataset(
            {
                "entered": (("step", "variant"), [[1000, 1000], [300, 330]]),
                "converted": (("step", "variant"), [[300, 330], [60, 75]]),
            },
            coords={"step": ["signup", "paid"], "variant": ["control", "treatment"]},
        )
        to_funnel_dataset(ds)
    """
    if isinstance(data, pd.DataFrame):
        return _validate(_from_stage_counts(data, stage_order))
    if isinstance(data, xr.Dataset):
        if stage_order is not None:
            raise ValueError(
                "stage_order applies to a DataFrame of stage counts. "
                f"The {Dim.STEP!r} coordinate of a Dataset already carries the order."
            )
        return _validate(data)
    raise TypeError(
        f"data must be a pandas DataFrame or an xarray Dataset; got {type(data).__name__}."
    )


def _from_stage_counts(
    data: pd.DataFrame, stage_order: Sequence[Hashable] | None
) -> xr.Dataset:
    """Turn a long table of stage counts into ``entered`` and ``converted`` per step."""
    missing = [col for col in (Col.STAGE, Col.COUNT) if col not in data.columns]
    if missing:
        raise ValueError(
            f"data must have the columns {Col.STAGE!r} and {Col.COUNT!r}; missing {missing}."
        )
    groups = [col for col in data.columns if col not in (Col.STAGE, Col.COUNT)]
    keys = [Col.STAGE, *groups]
    unlabelled = [col for col in keys if data[col].isna().any()]
    if unlabelled:
        raise ValueError(
            f"The columns {unlabelled} have rows without a label. Every row needs "
            "a stage and a label for every group column."
        )
    duplicated = data.duplicated(keys)
    if duplicated.any():
        cell = data.loc[duplicated, keys].iloc[0].to_dict()
        raise ValueError(
            f"data has more than one row for {_format_cell(cell)}. "
            f"The columns {groups} are read as groups, so aggregate the counts "
            "or drop the columns that are not groups."
        )

    order = _stage_order(data[Col.STAGE], stage_order)
    counts = data.set_index(keys)[Col.COUNT].to_xarray().sel({Col.STAGE: order})
    absent = counts.isnull()
    if absent.any():
        raise ValueError(
            f"data has no count for {_format_cell(_first_cell(absent))}. "
            f"Every group needs one count per stage; the columns {groups} are read as groups."
        )
    counts = _whole_counts(counts, Col.COUNT)

    increase = counts.diff(Col.STAGE) > 0
    if increase.any():
        cell = _first_cell(increase)
        current = cell.pop(Col.STAGE)
        previous = order[order.index(current) - 1]
        where = f" for {_format_cell(cell)}" if cell else ""
        raise ValueError(
            f"The count increases from stage {previous!r} to stage {current!r}{where}. "
            "A closed funnel can only lose units along the way. Check stage_order, "
            "and check that no unit joins the funnel after the first stage."
        )

    entered = (
        counts.isel({Col.STAGE: slice(None, -1)})
        .assign_coords({Col.STAGE: order[1:]})
        .rename({Col.STAGE: Dim.STEP})
    )
    converted = counts.isel({Col.STAGE: slice(1, None)}).rename({Col.STAGE: Dim.STEP})
    return xr.Dataset({Var.ENTERED: entered, Var.CONVERTED: converted})


def _stage_order(
    stage: pd.Series, stage_order: Sequence[Hashable] | None
) -> list[Hashable]:
    """Return the stages from first to last, checked against the stage column."""
    if stage_order is not None:
        order = list(stage_order)
    elif isinstance(stage.dtype, pd.CategoricalDtype) and stage.cat.ordered:
        order = list(stage.cat.remove_unused_categories().cat.categories)
    else:
        raise ValueError(
            "The order of the stages is unknown. Pass stage_order, or make the "
            f"{Col.STAGE!r} column an ordered categorical."
        )
    if len(set(order)) != len(order):
        raise ValueError(f"stage_order repeats a stage: {order}.")

    labels = set(stage.unique())
    not_in_data = [label for label in order if label not in labels]
    not_in_order = sorted(labels.difference(order), key=str)
    if not_in_data or not_in_order:
        raise ValueError(
            f"stage_order and the {Col.STAGE!r} column differ. "
            f"In stage_order but not in the data: {not_in_data}. "
            f"In the data but not in stage_order: {not_in_order}."
        )
    if len(order) < 2:
        raise ValueError(f"A funnel needs at least two stages; got {order}.")
    return order


def _validate(ds: xr.Dataset) -> xr.Dataset:
    """Check ``entered`` and ``converted`` and return them as integers, step first."""
    missing = [name for name in (Var.ENTERED, Var.CONVERTED) if name not in ds]
    if missing:
        raise ValueError(
            f"The dataset must have the variables {Var.ENTERED!r} and "
            f"{Var.CONVERTED!r}; missing {missing}."
        )
    entered, converted = ds[Var.ENTERED], ds[Var.CONVERTED]
    if set(entered.dims) != set(converted.dims):
        raise ValueError(
            f"{Var.ENTERED!r} and {Var.CONVERTED!r} must have the same dimensions; "
            f"got {entered.dims} and {converted.dims}."
        )
    if Dim.STEP not in converted.dims:
        raise ValueError(
            f"{Var.ENTERED!r} and {Var.CONVERTED!r} need a {Dim.STEP!r} dimension; "
            f"got {converted.dims}."
        )
    for dim in converted.dims:
        if dim not in ds.coords:
            raise ValueError(
                f"The dimension {dim!r} has no coordinate. Label it with "
                "Dataset.assign_coords so that the results can be read."
            )
        if not isinstance(ds[dim].dtype, np.dtype):
            # Labels with a pandas extension dtype, such as a categorical,
            # cannot be written to netCDF when the model is saved.
            ds = ds.assign_coords({dim: np.asarray(ds[dim].values)})

    dims = (Dim.STEP, *(dim for dim in converted.dims if dim != Dim.STEP))
    entered = _whole_counts(ds[Var.ENTERED], Var.ENTERED).transpose(*dims)
    converted = _whole_counts(ds[Var.CONVERTED], Var.CONVERTED).transpose(*dims)
    excess = converted > entered
    if excess.any():
        raise ValueError(
            f"{Var.CONVERTED!r} has more conversions than entrants in "
            f"{int(excess.sum())} cell(s). The first one is "
            f"{_format_cell(_first_cell(excess))}."
        )
    later = entered.isel({Dim.STEP: slice(1, None)})
    broken = later != converted.isel({Dim.STEP: slice(None, -1)}).values
    if broken.any():
        raise ValueError(
            f"The data do not describe a closed funnel: in {int(broken.sum())} cell(s) "
            f"{Var.ENTERED!r} differs from {Var.CONVERTED!r} at the step before. "
            f"The first one is {_format_cell(_first_cell(broken))}. The units that "
            "enter a step must be the ones that converted at the previous step."
        )
    return ds.assign({Var.ENTERED: entered, Var.CONVERTED: converted})


def _whole_counts(values: xr.DataArray, name: str) -> xr.DataArray:
    """Return ``values`` as ``int64`` after checking they are counts."""
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name!r} must be numeric; got dtype {values.dtype}.")
    if values.isnull().any():
        raise ValueError(f"{name!r} has missing values.")
    if (values < 0).any():
        raise ValueError(f"{name!r} has negative values.")
    if (values != np.floor(values)).any():
        raise ValueError(f"{name!r} must hold whole numbers.")
    return values.astype("int64")


def _first_cell(mask: xr.DataArray) -> dict[Hashable, Any]:
    """Return the coordinate labels of the first ``True`` cell of ``mask``."""
    position = np.argwhere(mask.values)[0]
    cell = {}
    for dim, index in zip(mask.dims, position, strict=True):
        label = mask[dim].values[index]
        cell[dim] = label.item() if isinstance(label, np.generic) else label
    return cell


def _format_cell(cell: dict[Hashable, Any]) -> str:
    """Format coordinate labels as ``dim='label', ...`` for an error message."""
    return ", ".join(f"{dim}={label!r}" for dim, label in cell.items())
