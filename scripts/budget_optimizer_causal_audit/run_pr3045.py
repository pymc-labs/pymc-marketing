"""Run the spend-dependent price example against PR #3045's checkout.

Example::

    PYTHONPATH=/path/to/pr3045/checkout python \
        scripts/budget_optimizer_causal_audit/run_pr3045.py
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr
from run import BUDGET, GRID, make_optimizer, solve

from pymc_marketing.mmm import PowerPriceResponse


def main() -> None:
    reference = xr.DataArray([2.0, 2.0], dims="channel", coords={"channel": ["A", "B"]})
    price_response = PowerPriceResponse(
        elasticity={"A": 0.5, "B": 0.0},
        reference_spend=reference,
        assume_delivery_units=True,
    )
    optimizer = make_optimizer((2.0, 1.0), n_dates=1, price_response=price_response)
    chosen = solve(optimizer)

    def truth(a, b):
        return 2 * np.log1p(np.sqrt(2 * a)) + np.log1p(b)

    values = [truth(a, BUDGET - a) for a in GRID]
    best = int(np.argmax(values))
    row = {
        "case": "spend_dependent_price_pr3045",
        "optimized_allocation": [round(v, 4) for v in chosen],
        "oracle_allocation": [
            round(float(GRID[best]), 4),
            round(float(BUDGET - GRID[best]), 4),
        ],
        "optimized_truth": round(float(truth(*chosen)), 6),
        "oracle_truth": round(float(values[best]), 6),
        "regret": round(max(0.0, float(values[best] - truth(*chosen))), 6),
    }
    if row["regret"] >= 0.001:
        raise RuntimeError("PR #3045 price-response allocation missed the oracle")
    Path(__file__).with_name("pr3045_results.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
