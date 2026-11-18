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

"""Shared noise summaries for experiment designs."""

import numpy as np


def residual_statistics(residuals: np.ndarray) -> tuple[float, float]:
    """Estimate the residual standard deviation and lag-one correlation."""
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Residuals must be nonempty and finite.")
    std = float(np.std(residuals))
    left, right = residuals[:-1], residuals[1:]
    if residuals.size <= 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return std, 0.0
    return std, float(np.corrcoef(left, right)[0, 1])
