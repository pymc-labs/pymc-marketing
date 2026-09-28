# PR: Explain how lift calibration informs marginal ROAS

Closes #3071

## Issue Summary

The lift-test notebooks showed changes in saturation curves and parameters without clearly connecting calibration to marginal ROAS, the quantity used to reason about incremental budget decisions.

## Root Cause

Both notebooks focused their result narratives on full curve and parameter recovery, which can imply that calibration should recover every part of a response curve or improve uncertainty uniformly. The new mROAS cells also used an outdated ArviZ HDI keyword and had no saved figure outputs.

## Solution

Explain how finite lift contrasts inform local mROAS through the fitted response curve. Add before-and-after posterior comparisons at matched spend, show 94% HDIs and simulated values, distinguish direct evidence in treated geos from hierarchical sharing in controls, and qualify claims about uncertainty, curve recovery, and applicability. Save the executed plot outputs in both notebooks.

## Changes Made

- `docs/source/notebooks/mmm/mmm_lift_test.ipynb`: Add mROAS comparisons for the uncalibrated model, calibrated model, and model with additional lift tests. In this simulation, posterior means move toward the simulated values; more tests narrow the intervals at the selected spends.
- `docs/source/notebooks/mmm/mmm_geolift_calibration.ipynb`: Add treated-versus-control geo mROAS comparison. In this simulation, many intervals narrow while posterior means remain below truth and often move farther away, illustrating that calibration does not guarantee better local mROAS accuracy.

## Testing

- [x] Both notebooks executed successfully with Papermill and saved the new plot outputs.
- [x] Commit hooks passed, including Ruff and notebook format validation.

## Notes

The mROAS plots evaluate the static saturation-curve slope in normalized simulation units. They do not represent a full campaign time path with adstock carryover.
