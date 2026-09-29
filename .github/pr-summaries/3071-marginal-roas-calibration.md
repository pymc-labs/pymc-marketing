# PR: Explain how lift calibration informs marginal ROAS

Closes #3071

## Issue Summary

Lift-test examples should connect experimental finite returns to the response curve's local marginal ROAS. The first comparison mixed scaled posterior parameters with unscaled simulated truth, so its apparent bias did not measure the same quantity.

## Solution

Report spend and sales in $ millions per week and mROAS in $ sales per $ spend. Convert posterior saturation parameters back to those business units for every parameter, curve, bias, and mROAS comparison. Check the model-implied finite lift against each supplied measurement before interpreting the point slope.

## Changes Made

- `docs/source/notebooks/mmm/mmm_lift_test.ipynb`: Treat the lift rows as noisy summaries of independent studies, correct the parameter and mROAS comparisons, align the plotted true adstock lag with the generating model, and show staged finite-lift and mROAS results.
- `docs/source/notebooks/mmm/mmm_geolift_calibration.ipynb`: Preserve an untreated spend schedule, estimate four geo lifts from observed treated and paired-control outcomes, and use matching late-test effective-spend contrasts after adstock has built up. Fit both MMMs only on pre-test observations, compare against correctly scaled simulated truth, and check finite lift alongside mROAS.

## Testing

- [x] Both notebooks executed successfully with Papermill; figures and tables are saved in the notebooks.
- [x] Ruff check and format, nbformat validation, and pre-commit passed.

## Notes

The national notebook simulates independent lift-study estimates and focuses on the calibration API; it does not run a national interrupted-time-series analysis. The geo notebook uses a compact paired-control estimator to demonstrate the measurement handoff; a production study needs stronger design diagnostics and uncertainty analysis. A finite lift constrains its tested spend interval directly, while point mROAS follows from the fitted response curve.
