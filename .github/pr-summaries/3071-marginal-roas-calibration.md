# PR: Explain how lift calibration informs marginal ROAS

Closes #3071

## Issue Summary

The lift-test notebooks showed changes in saturation curves and parameters without clearly connecting calibration to marginal ROAS, the quantity used to reason about incremental budget decisions.

## Root Cause

Both notebooks focused their result narratives on full curve and parameter recovery, which can imply that calibration should recover every part of a response curve or improve uncertainty uniformly.

## Solution

Add a short explanation of the link between finite lift contrasts, fitted response curves, and local mROAS. Lead the comparison sections with before-and-after posterior mROAS at matched baseline spend, show 94% HDIs and simulated values, and distinguish direct evidence in treated geos from hierarchical sharing in controls. Qualify claims about curve recovery, uncertainty, spend range, and carryover.

## Changes Made

- `docs/source/notebooks/mmm/mmm_lift_test.ipynb`: Add mROAS comparisons for the uncalibrated model, calibrated model, and model with additional lift tests.
- `docs/source/notebooks/mmm/mmm_geolift_calibration.ipynb`: Add treated-versus-control geo mROAS comparison and align result framing with direct and hierarchical evidence.

## Testing

- [ ] Notebook execution and sampling not run.
- [x] Notebook JSON edits reviewed and `git diff --check` passed.

## Notes

The mROAS plots evaluate the static saturation-curve slope in normalized simulation units. They do not represent a full campaign time path with adstock carryover.
