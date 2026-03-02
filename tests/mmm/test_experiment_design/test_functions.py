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

"""Tests for lightweight numpy response functions."""

import numpy as np
import pytest

from pymc_marketing.mmm.experiment_design.functions import logistic_saturation


class TestLogisticSaturation:
    """Tests for the numpy logistic saturation function."""

    def test_zero_input(self):
        result = logistic_saturation(0.0, lam=1.0, beta=1.0)
        assert result == pytest.approx(0.0)

    def test_large_input_approaches_beta(self):
        result = logistic_saturation(100.0, lam=1.0, beta=5.0)
        assert result == pytest.approx(5.0, abs=0.01)

    @pytest.mark.parametrize(
        "x,lam,beta,expected",
        [
            (1.0, 1.0, 1.0, (1 - np.exp(-1)) / (1 + np.exp(-1))),
            (2.0, 0.5, 3.0, 3.0 * (1 - np.exp(-1)) / (1 + np.exp(-1))),
            (0.5, 2.0, 2.0, 2.0 * (1 - np.exp(-1)) / (1 + np.exp(-1))),
        ],
    )
    def test_known_values(self, x, lam, beta, expected):
        result = logistic_saturation(x, lam, beta)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_monotonically_increasing(self):
        x = np.linspace(0, 10, 100)
        y = logistic_saturation(x, lam=1.0, beta=1.0)
        assert np.all(np.diff(y) > 0)

    def test_broadcast_1d(self):
        x = np.array([0.0, 1.0, 2.0])
        lam = np.array([1.0, 1.0, 1.0])
        beta = np.array([2.0, 2.0, 2.0])
        result = logistic_saturation(x, lam, beta)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0)

    def test_broadcast_2d(self):
        x = np.ones((4, 3))
        lam = np.ones((4, 1))
        beta = np.ones((1, 3))
        result = logistic_saturation(x, lam, beta)
        assert result.shape == (4, 3)

    def test_negative_input(self):
        result_pos = logistic_saturation(1.0, lam=1.0, beta=1.0)
        result_neg = logistic_saturation(-1.0, lam=1.0, beta=1.0)
        assert result_neg == pytest.approx(-result_pos)

    def test_higher_lam_steeper(self):
        x = 0.5
        y_low = logistic_saturation(x, lam=0.5, beta=1.0)
        y_high = logistic_saturation(x, lam=2.0, beta=1.0)
        assert y_high > y_low
