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
"""Synthetic data generators for Marketing Mix Models.

Each module simulates one kind of market together with its answer key:

- :mod:`~pymc_marketing.mmm.synthetic_data.endogenous_spend`: markets in
  which media spend responds to unrecorded demand.
"""

from pymc_marketing.mmm.synthetic_data.endogenous_spend import (
    EndogenousSpendMarket,
    simulate_endogenous_spend_market,
    tv_marginal_return,
)

__all__ = [
    "EndogenousSpendMarket",
    "simulate_endogenous_spend_market",
    "tv_marginal_return",
]
