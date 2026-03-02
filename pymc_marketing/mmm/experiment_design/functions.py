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

"""Lightweight numpy response functions for experiment design.

These are numpy equivalents of the PyTensor functions used during model
fitting. They support broadcasting for vectorised evaluation over
posterior draws.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def logistic_saturation(
    x: npt.NDArray[np.floating],
    lam: npt.NDArray[np.floating],
    beta: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Logistic saturation with scale parameter.

    Computes ``beta * (1 - exp(-lam * x)) / (1 + exp(-lam * x))``.

    Supports numpy broadcasting: ``x``, ``lam``, and ``beta`` can be
    scalars, 1-D arrays (posterior draws), or 2-D arrays
    (draws x time steps).

    Parameters
    ----------
    x : array_like
        Input (adstocked spend).
    lam : array_like
        Efficiency parameter (controls steepness).
    beta : array_like
        Scale parameter (maximum contribution).

    Returns
    -------
    np.ndarray
        Saturated response values.
    """
    x = np.asarray(x, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    exp_term = np.exp(-lam * x)
    return beta * (1.0 - exp_term) / (1.0 + exp_term)
