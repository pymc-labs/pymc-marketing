# PR: Explain how lift calibration informs marginal ROAS

Closes #3071

## Issue Summary

Lift-test examples should connect experimental finite returns to the response curve's local marginal ROAS. The first comparison mixed scaled posterior parameters with unscaled simulated truth, so its apparent bias did not measure the same quantity.

## Solution

Report spend and sales in $ millions per week and mROAS in $ sales per $ spend. Use fixed unit scaling across simulation and fitting, so new historical maxima cannot shift parameter coordinates during a refresh. Compare model-implied finite lift with each supplied measurement before interpreting the point slope.

## Changes Made

- `docs/source/notebooks/mmm/mmm_lift_test.ipynb`: Treat the lift rows as noisy summaries of independent studies, correct the parameter and mROAS comparisons, align the plotted true adstock lag with the generating model, and show staged finite-lift and mROAS results.
- `docs/source/notebooks/mmm/mmm_geolift_calibration.ipynb`: Preserve an untreated spend schedule, estimate four geo lifts from observed treated and paired-control outcomes, and use matching late-test effective-spend contrasts after adstock has built up. Fit both MMMs only on pre-test observations, compare against correctly scaled simulated truth, and check finite lift alongside mROAS.
- Both notebooks place triangle anchors and vertical sides on the known simulated saturation function at the programmed effective spends. Red circles with ±1 standard-error bars show noisy measured lift. The notebooks report mROAS absolute error and interval width separately.
- Both notebooks now call out why scaling constants must remain fixed as history grows: new data maxima otherwise shift internal parameter coordinates and the business-unit meaning of priors.
- Replace a fragile four-standard-error posterior-mean assertion with a standardized finite-lift discrepancy table. The assertion caused remote docs jobs to fail under mocked sampling even when the notebook calculations were valid.

## Testing

- [x] Both notebooks executed successfully with Papermill; figures and tables are saved in the notebooks.
- [x] The CI-style mocked notebook runner passed for both notebooks.
- [x] Ruff check and format, nbformat validation, and pre-commit passed.

## Notes

The national notebook simulates independent lift-study estimates and focuses on the calibration API; it does not run a national interrupted-time-series analysis. The geo notebook uses a compact paired-control estimator to demonstrate the measurement handoff; a production study needs stronger design diagnostics and uncertainty analysis. A finite lift constrains its tested spend interval directly, while point mROAS follows from the fitted response curve.

In the saved results, the expanded national lift set improves mROAS absolute error and 94% interval width for both channels relative to M0, though the first set has mixed effects. Geo calibration improves mROAS point bias in three of four treated markets and narrows all four intervals. `geo_00` is the exception: its noisy controlled lift estimate is substantially below the known simulated effect.
