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
"""Diagnostics namespace — posterior/prior predictive and residual plots."""

from __future__ import annotations

from typing import Any

import arviz as az
import arviz_plots as azp
import numpy as np
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting._helpers import (
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)


def _get_posterior_predictive(data: MMMIDataWrapper) -> xr.Dataset:
    """Return the posterior_predictive group from *data*.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding the fitted model's InferenceData.

    Returns
    -------
    xr.Dataset
        The posterior_predictive group.

    Raises
    ------
    ValueError
        If posterior_predictive is absent from idata.
    """
    if (
        not hasattr(data.idata, "posterior_predictive")
        or data.idata.posterior_predictive is None
    ):
        raise ValueError(
            "No posterior_predictive data found in idata. "
            "Run MMM.sample_posterior_predictive() first."
        )
    return data.idata.posterior_predictive


def _get_prior_predictive(data: MMMIDataWrapper) -> xr.Dataset:
    """Return the prior_predictive group from *data*.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding the fitted model's InferenceData.

    Returns
    -------
    xr.Dataset
        The prior_predictive group.

    Raises
    ------
    ValueError
        If prior_predictive is absent from idata.
    """
    if (
        not hasattr(data.idata, "prior_predictive")
        or data.idata.prior_predictive is None
    ):
        raise ValueError(
            "No prior_predictive data found in idata. "
            "Run MMM.sample_prior_predictive() first."
        )
    return data.idata.prior_predictive


def _get_prior(data: MMMIDataWrapper) -> xr.Dataset:
    """Return the prior group from *data*.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding the fitted model's InferenceData.

    Returns
    -------
    xr.Dataset
        The prior group.

    Raises
    ------
    ValueError
        If prior is absent from idata.
    """
    if not hasattr(data.idata, "prior") or data.idata.prior is None:
        raise ValueError(
            "No prior data found in idata. Run MMM.sample_prior_predictive() first."
        )
    return data.idata.prior


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
        return _get_prior(data)
    return _get_prior_predictive(data)


class DiagnosticsPlots:
    """Time-series diagnostic plots for fitted MMM models.

    Provides four methods to visualize model fit and residuals:

    - ``posterior_predictive`` — Posterior predictive time series with HDI.
    - ``prior_predictive``    — Prior predictive time series with HDI.
    - ``residuals``            — Residuals (target − predictions) over time.
    - ``residuals_distribution`` — Posterior distribution of residuals.

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
        pp_var: str = "y_original_scale",
    ) -> xr.DataArray:
        """Compute residuals as target_data - posterior predictions.

        Parameters
        ----------
        data : MMMIDataWrapper
            Wrapper holding idata with posterior_predictive and constant_data.
        pp_var : str, default "y_original_scale"
            Variable in posterior_predictive to use as predictions.

        Returns
        -------
        xr.DataArray
            Residuals named "residuals" with same dims as *pp_var*
            (typically ``(chain, draw, date[, extra_dims])``).

        Raises
        ------
        ValueError
            If *pp_var* not in posterior_predictive, or target_data not in constant_data.
        """
        pp_ds = _get_posterior_predictive(data)
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

        pp_ds = _get_posterior_predictive(data)

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
            layout_ds, cols=extra_dims, backend=backend, col_wrap=1, **pc_kwargs
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
    ) -> PlotCollection | tuple[Figure, NDArray[Axes]]:
        """Plot the posterior distribution of residuals using arviz-plots.

        Uses ``azp.plot_dist`` (KDE) with quantile reference lines via
        ``azp.add_lines``.  The distribution is computed over
        ``["chain", "draw", "date"]`` plus any dimensions in *aggregation*,
        so extra model dims (e.g. ``"geo"``) are structural facet dims by default.

        Parameters
        ----------
        quantiles : list[float], optional
            Quantile probabilities to mark as vertical reference lines.
            Default ``[0.0275, 0.5, 0.975]``. Each value must be in ``[0, 1]``.
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
            quantiles = [0.0275, 0.5, 0.975]
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

        if return_as_pc:
            return pc

        # azp.plot_dist creates a PlotCollection whose viz dataset does not
        # include a "plot" key (unlike PlotCollection.wrap).  Extract axes
        # directly from the figure instead.
        fig = pc.viz.ds["figure"].item()
        axes = np.atleast_1d(np.array(fig.get_axes()))
        return fig, axes
