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
"""Diagnostics namespace — posterior/prior predictive and residual plots.

This module exposes :class:`DiagnosticsPlots`, which is the entry point for all
time-series diagnostic plots on a fitted MMM model.  It is accessed via
``mmm.plot.diagnostics`` (a :class:`DiagnosticsPlots` instance backed by the
model's :class:`~pymc_marketing.data.idata.MMMIDataWrapper`).

Examples
--------
Construct the plotter directly from the model's data wrapper:

.. code-block:: python

    from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots

    dp = DiagnosticsPlots(mmm.data)

**Posterior predictive** — overlay posterior mean, HDI band, and observed target:

.. code-block:: python

    fig, axes = dp.posterior_predictive()

    # Scaled units, narrower HDI, two columns per row
    fig, axes = dp.posterior_predictive(original_scale=False, hdi_prob=0.8, col_wrap=2)

**Prior predictive** — same layout drawn from the prior:

.. code-block:: python

    fig, axes = dp.prior_predictive()

    fig, axes = dp.prior_predictive(original_scale=False, hdi_prob=0.8, col_wrap=2)

**Residuals over time** — mean residual line with HDI band and a zero reference:

.. code-block:: python

    fig, axes = dp.residuals_over_time()

    # Subset to one geo, custom styling
    fig, axes = dp.residuals_over_time(
        hdi_prob=0.8,
        dims={"geo": "geo_a"},
        line_kwargs={"color": "red", "linestyle": "-."},
        hdi_kwargs={"color": "red"},
    )

**Residuals distribution** — KDE of the posterior residual distribution:

.. code-block:: python

    fig, axes = dp.residuals_distribution(figsize=(8, 4))

    # Custom quantile markers, collapse geo into one distribution
    fig, axes = dp.residuals_distribution(
        figsize=(8, 4),
        quantiles=[0.1, 0.5, 0.9],
        aggregation="geo",
    )

**Posterior** — 1-D marginal KDE for selected variables:

.. code-block:: python

    fig, axes = dp.posterior(["saturation_lam", "adstock_alpha"], figsize=(10, 8))

**Prior vs posterior** — overlaid prior and posterior KDEs:

.. code-block:: python

    fig, axes = dp.prior_vs_posterior(
        ["saturation_lam", "adstock_alpha"], figsize=(10, 8)
    )
"""

from __future__ import annotations

from typing import Any

import arviz as az
import arviz_plots as azp
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.data.idata.utils import (
    get_posterior_predictive,
    get_prior,
    get_prior_predictive,
)
from pymc_marketing.mmm.plotting._helpers import (
    _dims_to_sel_kwargs,
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)


def _get_prior_for_plot(data: MMMIDataWrapper, original_scale: bool) -> xr.Dataset:
    """Return the correct idata group for prior predictive plotting.

    PyMC stores observed variables in ``idata.prior_predictive`` and
    Deterministics (such as ``y_original_scale``) in ``idata.prior``.
    This helper selects the right group based on *original_scale*.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding the model's InferenceData.
    original_scale : bool
        If True, return ``idata.prior`` (contains ``y_original_scale``).
        If False, return ``idata.prior_predictive`` (contains ``y``).

    Returns
    -------
    xr.Dataset
    """
    if original_scale:
        return get_prior(data.idata)
    return get_prior_predictive(data.idata)


class DiagnosticsPlots:
    """Time-series diagnostic plots for fitted MMM models.

    Provides six methods to visualize model fit and residuals:

    - ``posterior_predictive``   — Posterior predictive time series with HDI.
    - ``prior_predictive``       — Prior predictive time series with HDI.
    - ``residuals``              — Residuals (target − predictions) over time.
    - ``residuals_distribution`` — Posterior distribution of residuals.
    - ``posterior``              — 1-D marginal KDE distributions of posterior variables.
    - ``prior_vs_posterior``     — Overlaid prior and posterior KDE distributions.

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's InferenceData.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data

    def _plot_predictive(
        self,
        data: MMMIDataWrapper,
        pp_ds: xr.Dataset,
        original_scale: bool,
        hdi_prob: float,
        dims: dict[str, Any] | None,
        backend: str | None,
        line_kwargs: dict[str, Any] | None,
        hdi_kwargs: dict[str, Any] | None,
        observed_kwargs: dict[str, Any] | None,
        y_label: str,
        **pc_kwargs,
    ) -> PlotCollection:
        """Shared PlotCollection builder for posterior/prior predictive plots.

        Renders one panel per extra dimension combination. Each panel shows:
        - A mean line + HDI band for the predictive variable.
        - The observed target as a black line for reference, in the same scale.

        Parameters
        ----------
        data : MMMIDataWrapper
            Resolved data wrapper (already accounting for any idata override).
        pp_ds : xr.Dataset
            The posterior_predictive or prior_predictive group.
        original_scale : bool
            If True, plots ``y_original_scale``; if False, plots ``y``.
            Also controls the scale of the observed target line.
        hdi_prob : float
            HDI probability mass.
        dims : dict[str, Any] | None
            Subset dimensions forwarded to ``_select_dims``.
        backend : str | None
            PlotCollection backend.
        line_kwargs : dict | None
            Extra kwargs for the predictive mean ``azp.visuals.line_xy`` call.
        hdi_kwargs : dict | None
            Extra kwargs for ``azp.visuals.fill_between_y``.
        observed_kwargs : dict | None
            Extra kwargs for the observed ``azp.visuals.line_xy`` call.
            Defaults give a solid black line labelled "Observed".
        y_label : str
            Y-axis label (e.g. "Posterior Predictive").
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        PlotCollection
        """
        var_name = "y_original_scale" if original_scale else "y"
        var_da = _select_dims(pp_ds[var_name], dims)
        extra_dims = list(data.custom_dims)
        mean_da = var_da.mean(dim=("chain", "draw"))
        hdi_da = var_da.azstats.hdi(hdi_prob)

        layout_ds = mean_da.isel(date=0, drop=True).to_dataset(name="y")
        pc = PlotCollection.wrap(
            layout_ds,
            cols=extra_dims,
            backend=backend,
            **{"col_wrap": 1, **(pc_kwargs or {})},
        )

        dates = var_da.coords["date"].values

        pc.map(
            azp.visuals.line_xy,
            x=dates,
            y=mean_da,
            **{"label": var_name, **(line_kwargs or {})},
        )
        pc.map(
            azp.visuals.fill_between_y,
            x=dates,
            y_bottom=hdi_da.sel(ci_bound="lower"),
            y_top=hdi_da.sel(ci_bound="upper"),
            **{"alpha": 0.2, **(hdi_kwargs or {})},
        )

        # Observed target — uses the same scale as the predictions so they align.
        observed = data.get_target(original_scale=original_scale)
        observed = _select_dims(observed, dims)
        pc.map(
            azp.visuals.line_xy,
            x=dates,
            y=observed,
            **{"label": "Observed", "color": "black", **(observed_kwargs or {})},
        )

        pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
        pc.map(azp.visuals.labelled_y, text=y_label, ignore_aes={"color"})
        pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

        return pc

    def _compute_residuals(
        self,
        data: MMMIDataWrapper,
    ) -> xr.DataArray:
        """Compute residuals as target_data - posterior predictions.

        Parameters
        ----------
        data : MMMIDataWrapper
            Wrapper holding idata with posterior_predictive and constant_data.

        Returns
        -------
        xr.DataArray
            Residuals named "residuals" with same dims as ``y_original_scale``
            (typically ``(chain, draw, date[, extra_dims])``).

        Raises
        ------
        ValueError
            If ``y_original_scale`` not in posterior_predictive, or target_data not in constant_data.
        """
        pp_var = "y_original_scale"
        pp_ds = get_posterior_predictive(data.idata)
        if pp_var not in pp_ds:
            raise ValueError(
                f"Variable '{pp_var}' not found in posterior_predictive. "
                f"Available: {list(pp_ds.data_vars)}"
            )
        predictions = pp_ds[pp_var]
        target = data.get_target(original_scale=True)
        residuals = target - predictions
        residuals.name = "residuals"
        return residuals

    def posterior_predictive(
        self,
        original_scale: bool = True,
        hdi_prob: float = 0.94,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        observed_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot time series from the posterior predictive distribution.

        Creates one panel per extra-dimension combination (e.g. one per geo
        for geo-segmented models). Each panel overlays the posterior mean line,
        an HDI band, and the observed target.

        Parameters
        ----------
        original_scale : bool, default True
            If True, plots ``y_original_scale`` from posterior_predictive and
            the observed target in original units.
            If False, plots ``y`` (internal model scale) and the observed target
            in the same scaled units.
        hdi_prob : float, default 0.94
            Probability mass of the HDI band.
        idata : az.InferenceData, optional
            Override instance data. Constructs a local MMMIDataWrapper for this
            call only — does not mutate ``self._data``.
        dims : dict[str, Any], optional
            Subset dimensions, e.g. ``{"geo": ["CA", "NY"]}``.
        figsize : tuple[float, float], optional
            Figure size injected into ``figure_kwargs``.
        backend : str, optional
            Rendering backend. Non-matplotlib backends require ``return_as_pc=True``.
        return_as_pc : bool, default False
            If True, return the PlotCollection instead of ``(Figure, NDArray[Axes])``.
        line_kwargs : dict, optional
            Forwarded to ``azp.visuals.line_xy`` for the predictive mean line.
        hdi_kwargs : dict, optional
            Forwarded to ``azp.visuals.fill_between_y`` for the HDI band.
        observed_kwargs : dict, optional
            Forwarded to ``azp.visuals.line_xy`` for the observed data line.
            Default: solid black line labelled "Observed".
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection

        Examples
        --------
        .. code-block:: python

            fig, axes = mmm.plot.diagnostics.posterior_predictive()
            fig, axes = mmm.plot.diagnostics.posterior_predictive(
                original_scale=False, hdi_prob=0.50, dims={"geo": ["CA"]}
            )
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        pp_ds = get_posterior_predictive(data.idata)

        var_name = "y_original_scale" if original_scale else "y"
        if var_name not in pp_ds:
            raise ValueError(
                f"Variable '{var_name}' not found in posterior_predictive. "
                f"Available: {list(pp_ds.data_vars)}"
            )

        pc = self._plot_predictive(
            data=data,
            pp_ds=pp_ds,
            original_scale=original_scale,
            hdi_prob=hdi_prob,
            dims=dims,
            backend=backend,
            line_kwargs=line_kwargs,
            hdi_kwargs=hdi_kwargs,
            observed_kwargs=observed_kwargs,
            y_label="Posterior Predictive",
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)

    def prior_predictive(
        self,
        original_scale: bool = True,
        hdi_prob: float = 0.94,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        observed_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot time series from the prior predictive distribution.

        Mirrors ``posterior_predictive`` but draws from the prior_predictive
        group. Each panel overlays the prior mean line, an HDI band, and the
        observed target for comparison.

        Parameters
        ----------
        original_scale : bool, default True
            If True, plots ``y_original_scale`` from ``idata.prior`` (where
            PyMC stores Deterministics) and the observed target in original units.
            If False, plots ``y`` from ``idata.prior_predictive`` (where PyMC
            stores observed variables) and the observed target in scaled units.
        hdi_prob : float, default 0.94
            Probability mass of the HDI band.
        idata : az.InferenceData, optional
            Override instance data.
        dims : dict[str, Any], optional
            Subset dimensions.
        figsize : tuple[float, float], optional
        backend : str, optional
        return_as_pc : bool, default False
        line_kwargs : dict, optional
            Forwarded to ``azp.visuals.line_xy`` for the predictive mean line.
        hdi_kwargs : dict, optional
            Forwarded to ``azp.visuals.fill_between_y``.
        observed_kwargs : dict, optional
            Forwarded to ``azp.visuals.line_xy`` for the observed data line.
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection

        Examples
        --------
        .. code-block:: python

            fig, axes = mmm.plot.diagnostics.prior_predictive()
            fig, axes = mmm.plot.diagnostics.prior_predictive(original_scale=False)
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        pp_ds = _get_prior_for_plot(data, original_scale)

        var_name = "y_original_scale" if original_scale else "y"
        group_name = "prior" if original_scale else "prior_predictive"
        if var_name not in pp_ds:
            raise ValueError(
                f"Variable '{var_name}' not found in {group_name}. "
                f"Available: {list(pp_ds.data_vars)}"
            )

        pc = self._plot_predictive(
            data=data,
            pp_ds=pp_ds,
            original_scale=original_scale,
            hdi_prob=hdi_prob,
            dims=dims,
            backend=backend,
            line_kwargs=line_kwargs,
            hdi_kwargs=hdi_kwargs,
            observed_kwargs=observed_kwargs,
            y_label="Prior Predictive",
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)

    def residuals_over_time(
        self,
        hdi_prob: float = 0.94,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        hdi_kwargs: dict[str, Any] | None = None,
        line_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot residuals (target − posterior predictions) over time.

        Computes residuals using ``y_original_scale`` from posterior_predictive
        and ``target_data`` from constant_data. One panel per extra-dimension
        combination. Each panel shows a mean residuals line, an HDI band,
        and a zero reference line.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass for the residual band.
        idata : az.InferenceData, optional
            Override instance data.
        dims : dict[str, Any], optional
            Subset dimensions.
        figsize : tuple[float, float], optional
        backend : str, optional
        return_as_pc : bool, default False
        hdi_kwargs : dict, optional
            Forwarded to ``azp.visuals.fill_between_y``.
        line_kwargs : dict, optional
            Forwarded to ``azp.visuals.line_xy`` for the mean residuals line.
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection

        Examples
        --------
        .. code-block:: python

            fig, axes = mmm.plot.diagnostics.residuals_over_time()
            fig, axes = mmm.plot.diagnostics.residuals_over_time(hdi_prob=0.50)
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        residuals_da = self._compute_residuals(
            data
        )  # (chain, draw, date[, extra_dims])
        residuals_da = _select_dims(residuals_da, dims)

        extra_dims = list(data.custom_dims)
        mean_da = residuals_da.mean(dim=("chain", "draw"))  # (date[, extra_dims])
        hdi_da = residuals_da.azstats.hdi(hdi_prob)  # (date[, extra_dims], ci_bound)

        layout_ds = mean_da.isel(date=0, drop=True).to_dataset(name="residuals")
        pc = PlotCollection.wrap(
            layout_ds,
            cols=extra_dims,
            backend=backend,
            **{"col_wrap": 1, **(pc_kwargs or {})},
        )

        dates = residuals_da.coords["date"].values

        pc.map(
            azp.visuals.fill_between_y,
            x=dates,
            y_bottom=hdi_da.sel(ci_bound="lower"),
            y_top=hdi_da.sel(ci_bound="upper"),
            **{
                "alpha": 0.3,
                "label": f"{100 * hdi_prob:.0f}% HDI",
                **(hdi_kwargs or {}),
            },
        )
        pc.map(
            azp.visuals.line_xy,
            x=dates,
            y=mean_da,
            **{"label": "Mean residuals", **(line_kwargs or {})},
        )
        # Draw zero reference line using a zero-filled DataArray — PlotCollection.map
        # does not pass `ax`, so axhline cannot be used directly.
        zero_da = xr.zeros_like(mean_da)
        pc.map(
            azp.visuals.line_xy,
            x=dates,
            y=zero_da,
            **{"linestyle": "--", "color": "black", "label": "zero"},
        )

        pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
        pc.map(
            azp.visuals.labelled_y, text="Target − Predictions", ignore_aes={"color"}
        )
        pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

        return _extract_matplotlib_result(pc, return_as_pc)

    def residuals_distribution(
        self,
        quantiles: list[float] | None = None,
        aggregation: list[str] | str | None = None,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        dist_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot the posterior distribution of residuals using arviz-plots.

        Uses ``azp.plot_dist`` (KDE) with quantile reference lines via
        ``azp.add_lines``.  The distribution is computed over
        ``["chain", "draw", "date"]`` plus any dimensions in *aggregation*,
        so extra model dims (e.g. ``"geo"``) are structural facet dims by default.

        Parameters
        ----------
        quantiles : list[float], optional
            Quantile probabilities to mark as vertical reference lines.
            Default ``[0.025, 0.5, 0.975]``. Each value must be in ``[0, 1]``.
        aggregation : list[str] or str, optional
            Extra custom dimension names to collapse into the distribution
            (added to ``sample_dims`` beyond ``["chain", "draw", "date"]``).
            A single string is accepted and treated as ``[aggregation]``.
            Example: ``aggregation="geo"`` or ``aggregation=["geo"]`` merges geo
            panels into one combined distribution. Default ``None`` — extra dims
            are structural facet dims.
        idata : az.InferenceData, optional
            Override instance data.
        dims : dict[str, Any], optional
            Subset dimensions applied before plotting.
        figsize : tuple[float, float], optional
            Figure size forwarded via ``figure_kwargs``.
        backend : str, optional
            Rendering backend (e.g. ``"matplotlib"``). Non-matplotlib backends
            require ``return_as_pc=True``.
        return_as_pc : bool, default False
            Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
        dist_kwargs : dict, optional
            Extra kwargs forwarded to ``azp.plot_dist``.
        **pc_kwargs
            Forwarded to ``azp.plot_dist`` (e.g. ``figure_kwargs``).

        Returns
        -------
        PlotCollection or tuple[Figure, NDArray[Axes]]

        Examples
        --------
        .. code-block:: python

            fig, axes = mmm.plot.diagnostics.residuals_distribution()
            fig, axes = mmm.plot.diagnostics.residuals_distribution(
                quantiles=[0.05, 0.5, 0.95], aggregation=["geo"]
            )
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )

        if quantiles is None:
            quantiles = [0.025, 0.5, 0.975]
        for q in quantiles:
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"Each quantile must be in [0, 1]; got {q}.")

        residuals_da = self._compute_residuals(data)
        residuals_da = _select_dims(residuals_da, dims)

        if isinstance(aggregation, str):
            aggregation = [aggregation]
        if aggregation is not None:
            for dim in aggregation:
                if dim not in residuals_da.dims:
                    raise ValueError(
                        f"Dimension '{dim}' in aggregation not found in residuals. "
                        f"Available: {list(residuals_da.dims)}"
                    )

        sample_dims = ["chain", "draw", "date"] + (aggregation or [])

        ds = residuals_da.to_dataset(name="residuals")
        ref_ds = residuals_da.quantile(quantiles, dim=sample_dims)

        pc_kwargs = _process_plot_params(
            figsize=figsize, backend=backend, return_as_pc=return_as_pc, **pc_kwargs
        )

        n_quantiles = len(quantiles)
        line_colors = ["gray"] * n_quantiles

        pc = azp.plot_dist(
            ds,
            kind="kde",
            var_names=["residuals"],
            sample_dims=sample_dims,
            backend=backend,
            **(dist_kwargs or {}),
            **pc_kwargs,
        )
        pc = azp.add_lines(
            pc,
            values=ref_ds,
            ref_dim="quantile",
            aes_by_visuals={"ref_line": ["color"]},
            color=line_colors,
        )

        return _extract_matplotlib_result(pc, return_as_pc)

    def posterior(
        self,
        var_names: list[str] | str | None = None,
        group: str = "posterior",
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        kind: str = "kde",
        visuals: dict[str, Any] | None = None,
        aes: dict[str, Any] | None = None,
        aes_by_visuals: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot 1-D marginal KDE distributions for one or more posterior variables.

        Thin wrapper around ``azp.plot_dist``.

        Parameters
        ----------
        var_names : list[str] | str | None, optional
            Variable(s) to plot. ``None`` plots all variables in *group*.
        group : str, default "posterior"
            InferenceData group to draw from. Use ``"prior"`` to quickly inspect
            the prior without calling ``prior_vs_posterior``.
        idata : az.InferenceData, optional
            Override instance data for this call only.
        dims : dict[str, Any], optional
            Coordinate filters, e.g. ``{"channel": ["tv", "radio"]}``.
        figsize : tuple[float, float], optional
            Figure size forwarded via ``figure_kwargs``.
        backend : str, optional
            Rendering backend. Non-matplotlib backends require ``return_as_pc=True``.
        return_as_pc : bool, default False
            If True, return the raw ``PlotCollection``.
        kind : str, default "kde"
            Plot kind forwarded to ``azp.plot_dist`` (e.g. ``"kde"``, ``"hist"``).
        visuals : dict, optional
            Forwarded to ``azp.plot_dist``.
        aes : dict, optional
            Forwarded to ``azp.plot_dist`` as an explicit keyword argument.
        aes_by_visuals : dict, optional
            Forwarded to ``azp.plot_dist``.
        **pc_kwargs
            Forwarded to ``azp.plot_dist``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection

        Examples
        --------
        .. code-block:: python

            fig, axes = mmm.plot.diagnostics.posterior()
            fig, axes = mmm.plot.diagnostics.posterior(
                var_names=["alpha"], dims={"channel": ["tv"]}
            )
        """
        idata_to_use = idata if idata is not None else self._data.idata

        if not hasattr(idata_to_use, group) or getattr(idata_to_use, group) is None:
            raise ValueError(f"No {group} group found in idata. Fit the model first.")

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )
        coords = _dims_to_sel_kwargs(dims)

        pc = azp.plot_dist(
            idata_to_use,
            kind=kind,
            var_names=var_names,
            group=group,
            coords=coords,
            visuals=visuals,
            aes_by_visuals=aes_by_visuals,
            backend=backend,
            **({"aes": aes} if aes is not None else {}),
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)

    def prior_vs_posterior(
        self,
        var_names: list[str] | str | None = None,
        kind: str = "kde",
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        visuals: dict[str, Any] | None = None,
        aes: dict[str, Any] | None = None,
        aes_by_visuals: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Overlay prior and posterior 1-D marginal KDE distributions.

        Thin wrapper around ``azp.plot_prior_posterior``, which handles
        the prior/posterior colour legend automatically.

        Parameters
        ----------
        var_names : list[str] | str | None, optional
            Variable(s) to plot. ``None`` plots all variables present in
            both groups.
        kind : str, default "kde"
            Plot kind forwarded to ``azp.plot_prior_posterior``.
        idata : az.InferenceData, optional
            Override instance data for this call only.
        dims : dict[str, Any], optional
            Coordinate filters, e.g. ``{"channel": ["tv"]}``.
        figsize : tuple[float, float], optional
            Figure size forwarded via ``figure_kwargs``.
        backend : str, optional
            Rendering backend. Non-matplotlib backends require ``return_as_pc=True``.
        return_as_pc : bool, default False
            If True, return the raw ``PlotCollection``.
        visuals : dict, optional
            Forwarded to ``azp.plot_prior_posterior``.
        aes : dict, optional
            Forwarded to ``azp.plot_prior_posterior`` as an explicit keyword argument.
        aes_by_visuals : dict, optional
            Forwarded to ``azp.plot_prior_posterior``.
        **pc_kwargs
            Forwarded to ``azp.plot_prior_posterior``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection

        Examples
        --------
        .. code-block:: python

            fig, axes = mmm.plot.diagnostics.prior_vs_posterior()
            fig, axes = mmm.plot.diagnostics.prior_vs_posterior(
                var_names=["alpha"], dims={"channel": ["tv"]}
            )
        """
        idata_to_use = idata if idata is not None else self._data.idata

        if not hasattr(idata_to_use, "prior") or idata_to_use.prior is None:
            raise ValueError(
                "No prior group found in idata. "
                "Run MMM.sample_prior_predictive() first."
            )
        if not hasattr(idata_to_use, "posterior") or idata_to_use.posterior is None:
            raise ValueError("No posterior group found in idata. Fit the model first.")

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )
        coords = _dims_to_sel_kwargs(dims)

        pc = azp.plot_prior_posterior(
            idata_to_use,
            kind=kind,
            var_names=var_names,
            coords=coords,
            visuals=visuals,
            aes_by_visuals=aes_by_visuals,
            backend=backend,
            **({"aes": aes} if aes is not None else {}),
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)
