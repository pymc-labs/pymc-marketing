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
r"""The label an incrementality result carries: which estimand it is.

:class:`~pymc_marketing.mmm.incrementality.Incrementality` returns several
quantities with the same dimensions: a total increment, the part of it that
does not go through a mediator, the part that does, ratios of each to spend.
An array of numbers does not say which one it is, so each result is returned
with an :class:`EstimandLabel` in its attributes.

The label is one JSON attribute, :data:`LABEL_ATTR`.  It states the request
(the quantity, the paths, the period, the spend factor) and what was held
fixed, and it is bound to the array by a digest of the array's values, dimension
labels and ``complete`` coordinate together with the label's own fields.

.. code-block:: python

    from pymc_marketing.mmm import EstimandLabel

    direct = mmm.incrementality.compute_incremental_contribution(
        frequency="all_time", paths="direct"
    )
    label = EstimandLabel.read(direct)
    label.paths  # "direct"
    label.held  # ("funnel_effect_contribution",)
    label.definition  # "Y(x) - Y(x', M(x))"

**Read a label before cutting a result.**  :meth:`EstimandLabel.read` verifies
the digest, so it accepts the array the module returned, a copy, a transpose,
the array under another name and the same array read back from a file.  It
refuses anything derived from it: a selection, a sort, a renamed dimension,
arithmetic, a reduction, a concatenation.  What xarray does with attributes
under those operations differs between versions, so a derived array may still
carry the attribute.  It then says what the array was computed from, and
``read`` refuses it.  Drop the attributes of anything derived
(``result.drop_attrs()``).

In the ``definition``, :math:`x` is actual spend, :math:`x'` the counterfactual
spend, :math:`Y` the response, :math:`M(x)` the value the model implies for the
mediators at actual spend, and :math:`S` channel spend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    computed_field,
    model_validator,
)

from pymc_marketing.data.idata.schema import Frequency

__all__ = ["COMPLETE", "LABEL_ATTR", "EstimandLabel", "PathEffect"]

PathEffect = Literal["total", "direct", "mediated"]
"""Which paths from spend to the response an increment follows.

``"total"`` is every path.  ``"direct"`` holds every mediator at the value the
model implies for it at actual spend, and ``"mediated"`` is the rest of the
total.
"""

LABEL_ATTR = "incrementality_label"
"""The one attribute a result's label is written under."""

COMPLETE = "complete"
"""Coordinate marking the periods whose carryover the fitted dates do not cut short."""

_Text = Annotated[str, StringConstraints(strict=True, min_length=1)]
_Factor = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
_Quantity = Literal[
    "incremental_contribution",
    "contribution_over_spend",
    "spend_over_contribution",
    "marginal_contribution_over_spend",
    "incremental_mediator",
    "spend_change",
    "induced_spend",
]
_AVERAGE_RATIOS = ("contribution_over_spend", "spend_over_contribution")
_RATIOS = (*_AVERAGE_RATIOS, "marginal_contribution_over_spend")


def _dimension_labels(array: xr.DataArray, dim: str) -> bytes:
    """Return one dimension's labels in a form a file round trip does not change.

    Dates enter as nanoseconds.  Every other label enters as its type and its
    text, so the number ``1`` and the text ``'1'`` are two labels.
    """
    if dim not in array.coords:
        return b"<unlabelled>"
    values = np.asarray(array.coords[dim].values)
    if np.issubdtype(values.dtype, np.datetime64):
        return b"date:" + values.astype("datetime64[ns]").astype("<i8").tobytes()
    return repr(
        [(type(value).__name__, str(value)) for value in values.tolist()]
    ).encode()


def _digest_v1(result: xr.DataArray, fields: dict) -> str:
    """Digest an array together with the label fields stored on it.

    Parameters
    ----------
    result : xr.DataArray
        The array the label is for.
    fields : dict
        The label's fields as they are stored, without the definition.

    Returns
    -------
    str
        A hexadecimal digest of the dimension names, the shape, the values, the
        labels of every dimension, the ``complete`` coordinate when there is
        one, and *fields*.  The dimensions are laid out by name first, so a
        transpose does not change it.
    """
    ordered = result.transpose(*sorted(map(str, result.dims)))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr((ordered.dims, ordered.shape)).encode())
    digest.update(np.ascontiguousarray(ordered.values, dtype="<f8").data)
    for dim in map(str, ordered.dims):
        digest.update(dim.encode())
        digest.update(_dimension_labels(ordered, dim))
    if COMPLETE in ordered.coords:
        mark = ordered.coords[COMPLETE]
        digest.update(repr((COMPLETE, mark.dims)).encode())
        digest.update(np.ascontiguousarray(mark.values, dtype=bool).data)
    digest.update(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


class EstimandLabel(BaseModel):
    """What one incrementality result answers for.

    Built by the public methods of
    :class:`~pymc_marketing.mmm.incrementality.Incrementality` and read back
    with :meth:`read`.  A label that contradicts itself cannot be built: the
    validator names every rule a set of fields breaks.

    Parameters
    ----------
    schema_version : int
        Version of the fields and of the digest rule.
    quantity : str
        What was computed: an increment in the response, one of the three
        ratios to spend, the movement of a mediator, a change in channel spend
        or the spend a mediator induces.
    scope : {"per_channel", "joint"}
        Whether each channel was moved on its own or all of them together.
    frequency : str
        The time aggregation asked for.
    start_date, end_date : str
        The resolved period, ISO 8601.
    counterfactual_spend_factor : float
        Factor applied to spend in the counterfactual.  For a marginal ratio,
        one plus the fractional increase.
    include_carryover : bool
        Whether carryover outside each period was counted.
    central_tendency : {"median", "mean"}, optional
        Central tendency of the predictions differenced.  Response results only.
    accounting_verified : bool, optional
        Whether the terms the increment is assembled from were checked to
        reproduce the whole move in the linear predictor.  ``False`` only for a
        total returned with the warning that the check could not run.  Response
        results only.
    paths : {"total", "direct", "mediated"}, optional
        Which paths the increment follows.  Response results only.
    held : tuple of str
        The model variables held at their factual value, sorted.  Empty for a
        total, and for a model with no mediator.
    held_at : {"factual"}, optional
        Where the mediators are held.  Set for ``direct`` and ``mediated``.
    induced_spend : tuple of str
        The spend mediators whose movement is in the result: in the denominator
        of a ratio, or summed in an ``induced_spend`` variable.
    induced_spend_units : str, optional
        The units those mediators were declared in.
    mediator, variable, units, is_spend : optional
        The declared name of a mediator, its model variable and how its
        movement is reported.  Mediator results only.
    """

    # Strict per field, not ``strict=True`` on the model: with the model-level
    # switch pydantic 2.1 refuses a JSON array for a tuple field, so a label
    # with something held would not read back.
    model_config = ConfigDict(frozen=True, extra="forbid")

    SCHEMA_VERSION: ClassVar[int] = 1
    MIGRATIONS: ClassVar[dict[int, Callable[[dict], dict]]] = {}
    """``{n: upgrade}``, turning the stored fields of version ``n`` into those of ``n + 1``."""
    DIGESTS: ClassVar[dict[int, Callable[[xr.DataArray, dict], str]]] = {1: _digest_v1}
    """The digest rule of each schema version.  A label is verified by the rule that wrote it."""

    schema_version: StrictInt = 1
    quantity: _Quantity
    scope: Literal["per_channel", "joint"]
    frequency: Frequency
    start_date: _Text
    end_date: _Text
    counterfactual_spend_factor: _Factor
    include_carryover: StrictBool
    central_tendency: Literal["median", "mean"] | None = None
    accounting_verified: StrictBool | None = None
    paths: PathEffect | None = None
    held: tuple[_Text, ...] = ()
    held_at: Literal["factual"] | None = None
    induced_spend: tuple[_Text, ...] = ()
    induced_spend_units: _Text | None = None
    mediator: _Text | None = None
    variable: _Text | None = None
    units: _Text | None = None
    is_spend: StrictBool | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        problems: list[str] = []
        if self.schema_version != type(self).SCHEMA_VERSION:
            problems.append(
                f"this code writes schema version {type(self).SCHEMA_VERSION}"
            )

        period = []
        for name in ("start_date", "end_date"):
            try:
                period.append(datetime.fromisoformat(getattr(self, name)))
            except ValueError:
                problems.append(f"'{name}' is an ISO 8601 date or date-time")
        if len(period) == 2 and period[0] > period[1]:
            problems.append("the period starts before it ends")

        of_mediator = self.quantity == "incremental_mediator"
        of_spend = self.quantity in ("spend_change", "induced_spend")
        is_path = self.paths in ("direct", "mediated")
        response_only = (self.paths, self.central_tendency, self.accounting_verified)
        if of_mediator:
            if None in (self.mediator, self.variable, self.is_spend):
                problems.append(
                    "a mediator result names its mediator, variable and is_spend"
                )
            if self.is_spend and self.units is None:
                problems.append("a spend mediator has units")
        elif (self.mediator, self.variable, self.units, self.is_spend) != (None,) * 4:
            problems.append("mediator fields belong to a mediator result")
        if of_mediator or of_spend:
            if response_only != (None, None, None):
                problems.append(
                    "only a response result has paths, a central tendency and "
                    "an accounting"
                )
        elif None in response_only:
            problems.append(
                "a response result has paths, a central tendency and an accounting"
            )

        if is_path and self.accounting_verified is False:
            problems.append("a direct or mediated result is never returned unverified")
        if is_path != (self.held_at is not None):
            problems.append(
                "'held_at' goes with a direct or mediated path, and only with one"
            )
        if self.held and not is_path:
            problems.append("only a direct or mediated path holds anything")
        for name in ("held", "induced_spend"):
            values = getattr(self, name)
            if list(values) != sorted(set(values)):
                problems.append(f"'{name}' is sorted and without repeats")

        if bool(self.induced_spend) != (self.induced_spend_units is not None):
            problems.append(
                "'induced_spend_units' goes with 'induced_spend', and only with it"
            )
        if self.quantity == "induced_spend":
            if not self.induced_spend:
                problems.append("induced spend names its mediators")
        elif self.induced_spend and (
            self.quantity not in _RATIOS or self.paths != "total"
        ):
            problems.append(
                "induced spend enters the denominator of a total ratio only"
            )

        if self.quantity in _RATIOS and self.scope != "per_channel":
            problems.append("a ratio is per channel")
        if self.quantity == "spend_over_contribution" and self.paths == "mediated":
            problems.append("spend over a mediated share is not offered")
        if (
            self.quantity == "marginal_contribution_over_spend"
            and self.counterfactual_spend_factor <= 1.0
        ):
            problems.append("a marginal ratio raises spend")
        if self.quantity in _AVERAGE_RATIOS and self.counterfactual_spend_factor != 0.0:
            problems.append("an average ratio removes spend")

        if problems:
            raise ValueError("Incoherent label: " + "; ".join(problems) + ".")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def definition(self) -> str:
        """The contrast in symbols, with the sign the result carries.

        Above a spend factor of one the result is counterfactual minus actual,
        and at or below one it is actual minus counterfactual, so a channel
        with a positive effect has a positive number either way.
        """
        raised = self.counterfactual_spend_factor > 1.0
        if self.quantity in ("incremental_mediator", "induced_spend"):
            return "M(x') - M(x)" if raised else "M(x) - M(x')"
        spend = "S(x') - S(x)" if raised else "S(x) - S(x')"
        if self.quantity == "spend_change":
            return spend
        contrasts: dict[tuple[PathEffect | None, bool], str] = {
            ("total", False): "Y(x) - Y(x')",
            ("total", True): "Y(x') - Y(x)",
            ("direct", False): "Y(x) - Y(x', M(x))",
            ("direct", True): "Y(x', M(x)) - Y(x)",
            ("mediated", False): "Y(x', M(x)) - Y(x')",
            ("mediated", True): "Y(x') - Y(x', M(x))",
        }
        # The validator leaves no response result without its paths.
        contrast = contrasts[(self.paths, raised)]
        if self.quantity == "incremental_contribution":
            return contrast
        if self.induced_spend:
            spend = f"{spend} + " + ("M(x') - M(x)" if raised else "M(x) - M(x')")
        if self.quantity == "spend_over_contribution":
            return f"[{spend}] / [{contrast}]"
        return f"[{contrast}] / [{spend}]"

    def attach(self, result: xr.DataArray) -> xr.DataArray:
        """Return *result* with this label as its only attribute.

        One attribute, so that an operation which merges the attributes of two
        results keeps the label whole or drops it whole.

        Parameters
        ----------
        result : xr.DataArray
            The array the label describes.

        Returns
        -------
        xr.DataArray
            A shallow copy of *result* whose attributes hold the label and
            nothing else.
        """
        fields = self.model_dump(mode="json", exclude={"definition"})
        digest = type(self).DIGESTS[self.schema_version](result, fields)
        labelled = result.copy(deep=False)
        labelled.attrs = {
            LABEL_ATTR: json.dumps(
                {**fields, "definition": self.definition, "digest": digest}
            )
        }
        return labelled

    @classmethod
    def read(cls, result: xr.DataArray) -> Self:
        """Read and verify the label of *result*.

        Parameters
        ----------
        result : xr.DataArray
            An array returned by
            :class:`~pymc_marketing.mmm.incrementality.Incrementality`, as
            returned or read back from a file.

        Returns
        -------
        EstimandLabel
            The label the array was issued with, upgraded to the current schema.

        Raises
        ------
        ValueError
            If *result* is not a ``DataArray``; if it carries no label, or one
            that is not this module's; if the label was written by a newer
            schema than this code knows; or if the array and the label do not
            belong together, because a value, a dimension label, the
            ``complete`` coordinate or a stored field differs from what the
            label was issued for.
        """
        if not isinstance(result, xr.DataArray):
            raise ValueError(
                "A label belongs to one array: read it from a variable of the "
                f"Dataset, not from the {type(result).__name__}."
            )
        if LABEL_ATTR not in result.attrs:
            raise ValueError(
                f"No {LABEL_ATTR!r} attribute: the array is not a result of the "
                "incrementality module, or it was derived from one."
            )
        try:
            payload = json.loads(result.attrs[LABEL_ATTR])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"The {LABEL_ATTR!r} attribute is not JSON text."
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(f"The {LABEL_ATTR!r} attribute is not a JSON object.")

        version = payload.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(f"The label has no usable schema version: {version!r}.")
        if version > cls.SCHEMA_VERSION:
            raise ValueError(
                f"The label was written with schema version {version} and this "
                f"code reads up to {cls.SCHEMA_VERSION}: upgrade pymc-marketing."
            )

        stored_digest = payload.pop("digest", None)
        # The stored text is for a reader of the raw attribute.  It is a
        # function of the fields, so it is regenerated and never compared.
        payload.pop("definition", None)
        # Verified as stored, by the rule of the version that wrote it, before
        # anything is migrated: a match means these are the fields that version
        # wrote, so an edited or truncated label never reaches a migration.
        if stored_digest != cls.DIGESTS[version](result, payload):
            raise ValueError(
                "The array and its label do not belong together: a value, a "
                "dimension name, a coordinate label or the 'complete' mark "
                "differs from the result the label was issued for, or the label "
                "was edited.  Read a label before selecting from, reordering, "
                "renaming the dimensions of or computing with a result."
            )
        for step in range(version, cls.SCHEMA_VERSION):
            if step not in cls.MIGRATIONS:
                raise ValueError(
                    f"No migration from label schema {step} to {step + 1}."
                )
            payload = cls.MIGRATIONS[step](payload)
        return cls.model_validate(payload)
