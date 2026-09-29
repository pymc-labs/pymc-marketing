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
"""Simulation study for ``BudgetModelEffect``.

Two questions, each answered over many simulated markets:

1. **Recovery.** In the ``"forecast"`` scenario (budgets follow an unrecorded
   demand forecast), how well does each way of using a lift test recover TV's
   true marginal return? Arms:

   * ``plain``: plain MMM, test weeks as data;
   * ``lift_history``: plain MMM on the pre-test history plus the lift
     likelihood (the experiment enters once);
   * ``lift_full``: plain MMM on the full series plus the lift likelihood
     (what users commonly do; the experiment enters twice);
   * ``budget_design``: budget model with the experiment's design;
   * ``budget_no_design``: budget model without the design, to separate the
     design from the extra likelihood;
   * ``budget_design_lag1``, ``budget_design_lag3``: budget model with the
     design and one or three lagged surprises in the control function.

2. **Calibration of the diagnostic, and the cost of the correction.** In the
   ``"observed_only"`` scenario spend is exogenous given the controls, so a
   plain MMM has nothing to correct and every ``gamma`` should be zero. How much
   accuracy does the budget model give up there, and how often does the 94%
   interval exclude zero (the false-alarm rate), with a correctly specified
   sales equation and with a misspecified one (logistic saturation fit to
   Michaelis-Menten data)?

Usage::

    python scripts/budget_model/simulation_study.py --n-seeds 50 --workers 4 --out study
    python scripts/budget_model/simulation_study.py --summarise-only --out study

Each fit is appended to ``<out>/results.csv`` as it finishes, so an interrupted
run resumes where it stopped. ``<out>/summary.md`` holds the tables.

Expected results
----------------
With the defaults (seeds 1000-1049; 500 draws, 1,000 tuning steps and 2 chains
per fit; nutpie), the study takes a couple of hours on 5 workers and gives the
tables below. The true marginal return is about 2. Coverage is of the 94%
interval, with an exact 95% binomial CI. The interval score is the Winkler
score (width plus 2/0.06 times the distance by which the interval misses the
truth; lower is better). Divergences are per fit, out of 1,000 draws.

Recovery of TV's marginal return when budgets chase demand (``"forecast"``)::

    arm                 coverage           mean err  abs err  width  score  div (median/p90)
    plain               0.04 [0.00, 0.14]   +2.18     2.18    1.94   42.2   0 / 0
    lift_history        0.30 [0.18, 0.45]   +1.06     1.10    1.46   16.1   0 / 1
    lift_full           0.24 [0.13, 0.38]   +1.05     1.09    1.33   17.1   0 / 1
    budget_no_design    0.80 [0.66, 0.90]   +1.56     1.63    5.48    8.8   0 / 4
    budget_design       0.64 [0.49, 0.77]   +0.85     1.05    2.74    9.0   0 / 3
    budget_design_lag1  0.82 [0.69, 0.91]   +0.51     0.81    2.83    4.7   0 / 2
    budget_design_lag3  0.88 [0.76, 0.95]   +0.47     0.79    2.83    4.7   0 / 3

Recovery when spend is exogenous given the controls (``"observed_only"``)::

    arm                         coverage           mean err  abs err  width  score
    plain                       0.78 [0.64, 0.88]   +0.22     0.67    2.31    3.4
    budget_design               0.72 [0.58, 0.84]   +0.48     0.94    2.98    6.5
    budget_design_lag1          0.76 [0.62, 0.87]   +0.55     0.96    3.08    5.5
    budget_design_lag3          0.86 [0.73, 0.94]   +0.59     1.01    3.14    5.9
    budget_no_design            0.76 [0.62, 0.87]   +1.99     2.18    7.35   13.6
    budget_design_misspecified  0.82 [0.69, 0.91]   +0.27     0.88    3.17    4.3

Share of markets where the 94% interval for TV's ``gamma`` excludes zero
(power in ``"forecast"``, false alarms in ``"observed_only"``; nominal 6%)::

    scenario       arm                         excludes 0          mean P(gamma > 0)
    forecast       budget_design               0.56 [0.41, 0.70]   0.92
    forecast       budget_design_lag1          0.56 [0.41, 0.70]   0.92
    forecast       budget_design_lag3          0.50 [0.36, 0.64]   0.92
    forecast       budget_no_design            0.08 [0.02, 0.19]   0.66
    observed_only  budget_design               0.12 [0.05, 0.24]   0.41
    observed_only  budget_design_lag1          0.14 [0.06, 0.27]   0.39
    observed_only  budget_design_lag3          0.16 [0.07, 0.29]   0.39
    observed_only  budget_no_design            0.14 [0.06, 0.27]   0.30
    observed_only  budget_design_misspecified  0.08 [0.02, 0.19]   0.48

Reading: every arm overstates TV when budgets chase demand. The lift likelihood
halves the plain MMM's bias, and counting the experiment twice does not help.
The budget model with the design and one lagged surprise cuts the bias to about
+0.5, covers the truth in 82% of markets, and has less than a third of the lift
likelihood's interval score; without the design, its coverage comes only from
intervals twice as wide. When there is nothing to correct, the same
configuration gives up about 0.3 in bias (interval score 3.4 to 5.5), so with a
design the correction is cheap insurance, and without one it is a poor bet.
With a design, ``gamma`` detects demand-chasing budgets in about half of
markets, and its false-alarm rate is roughly twice nominal, with a negative
lean: an interval that excludes zero is evidence worth following up, not a
verdict. The untested Digital channel's false-alarm rate rises to 0.22 with
three lags, one reason to keep ``surprise_lags`` small.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

INTERVAL_PROB = 0.94
RECOVERY_ARMS = [
    "plain",
    "lift_history",
    "lift_full",
    "budget_design",
    "budget_no_design",
    "budget_design_lag1",
    "budget_design_lag3",
]
CALIBRATION_ARMS = [
    "plain",
    "budget_design",
    "budget_no_design",
    "budget_design_misspecified",
    "budget_design_lag1",
    "budget_design_lag3",
]
COLUMNS = [
    "scenario",
    "arm",
    "seed",
    "truth",
    "mean",
    "lower",
    "upper",
    "divergences",
    *[
        f"gamma_{channel}_{stat}"
        for channel in ("tv", "digital")
        for stat in ("lower", "upper", "prob_positive")
    ],
]


def _make_mmm(saturation: str = "michaelis_menten"):
    from pymc_marketing.mmm import (
        MMM,
        GeometricAdstock,
        LogisticSaturation,
        MichaelisMentenSaturation,
    )

    return MMM(
        date_column="date",
        channel_columns=["tv", "digital"],
        control_columns=["inflation", "unemployment"],
        target_column="y",
        yearly_seasonality=4,
        adstock=GeometricAdstock(l_max=8),
        saturation=(
            MichaelisMentenSaturation()
            if saturation == "michaelis_menten"
            else LogisticSaturation()
        ),
    )


def _tv_marginal_return(mmm, x: float) -> np.ndarray:
    """Marginal return of TV at steady spend ``x``, in original units."""
    from pytensor.xtensor import as_xtensor

    posterior = mmm.idata.posterior
    scale_x = float(mmm.scalers["_channel"].sel(channel="tv"))
    scale_y = float(mmm.scalers["_target"])
    params = {
        name.removeprefix("saturation_"): as_xtensor(
            posterior[name].sel(channel="tv").to_numpy().ravel(), dims=("sample",)
        )
        for name in posterior.data_vars
        if name.startswith("saturation_") and "channel" in posterior[name].dims
    }
    # Central difference of the fitted curve, vectorised over posterior draws.
    eps = 1e-3
    xs = as_xtensor(np.array([x - eps, x + eps]) / scale_x, dims=("point",))
    values = mmm.saturation.function(xs, **params).transpose("point", "sample").eval()
    return (values[1] - values[0]) * scale_y / (2 * eps)


def _fit_one(scenario: str, arm: str, seed: int, sample_kwargs: dict) -> dict:
    warnings.filterwarnings("ignore")
    from pymc_marketing.mmm import BudgetModelEffect
    from pymc_marketing.mmm.synthetic_data import simulate_endogenous_spend_market

    market = simulate_endogenous_spend_market(scenario, random_seed=seed)
    data, n_hist = market.data, market.n_history
    X, y = data.drop(columns="y"), data["y"]
    effect = None
    mmm = _make_mmm("logistic" if arm.endswith("misspecified") else "michaelis_menten")

    if arm == "lift_history":
        mmm.build_model(X.iloc[:n_hist], y.iloc[:n_hist])
        mmm.add_lift_test_measurements(market.lift_test)
        mmm.fit(X.iloc[:n_hist], y.iloc[:n_hist], random_seed=seed, **sample_kwargs)
    elif arm == "lift_full":
        mmm.build_model(X, y)
        mmm.add_lift_test_measurements(market.lift_test)
        mmm.fit(X, y, random_seed=seed, **sample_kwargs)
    else:
        if arm.startswith("budget"):
            design = None if arm == "budget_no_design" else market.design
            lags = int(arm.rsplit("lag", 1)[1]) if "_lag" in arm else 0
            effect = BudgetModelEffect(design=design, surprise_lags=lags)
            mmm.add_mu_effect(effect)
        mmm.fit(X, y, random_seed=seed, **sample_kwargs)

    values = _tv_marginal_return(mmm, market.operating_point)
    tail = (1 - INTERVAL_PROB) / 2
    lower, upper = np.quantile(values, [tail, 1 - tail])
    truth = market.true_marginal_return
    row = {
        "scenario": scenario,
        "arm": arm,
        "seed": seed,
        "truth": truth,
        "mean": values.mean(),
        "lower": lower,
        "upper": upper,
        "divergences": int(mmm.idata.sample_stats["diverging"].sum()),
    }
    if effect is not None:
        summary = effect.exogeneity_summary(mmm, interval_prob=INTERVAL_PROB)
        for channel, frame in summary.groupby("channel"):
            row[f"gamma_{channel}_lower"] = float(frame["gamma_lower"].iloc[0])
            row[f"gamma_{channel}_upper"] = float(frame["gamma_upper"].iloc[0])
            row[f"gamma_{channel}_prob_positive"] = float(
                frame["prob_positive"].iloc[0]
            )
    return row


def _jobs(n_seeds: int, first_seed: int) -> list[tuple[str, str, int]]:
    seeds = range(first_seed, first_seed + n_seeds)
    return [("forecast", arm, s) for s in seeds for arm in RECOVERY_ARMS] + [
        ("observed_only", arm, s) for s in seeds for arm in CALIBRATION_ARMS
    ]


def _binomial_ci(successes: int, n: int) -> tuple[float, float]:
    from scipy.stats import binomtest

    if n == 0:
        return (np.nan, np.nan)
    ci = binomtest(successes, n).proportion_ci(confidence_level=0.95, method="exact")
    return (ci.low, ci.high)


def _interval_score(frame: pd.DataFrame) -> pd.Series:
    """Winkler interval score: width plus a penalty for missing the truth."""
    alpha = 1 - INTERVAL_PROB
    below = (frame["lower"] - frame["truth"]).clip(lower=0)
    above = (frame["truth"] - frame["upper"]).clip(lower=0)
    return (frame["upper"] - frame["lower"]) + (2 / alpha) * (below + above)


def summarise(results: pd.DataFrame) -> str:
    """Render the recovery and calibration tables as markdown."""
    lines = []
    titles = {
        "forecast": (
            "## Recovery of TV's marginal return (forecast scenario)",
            RECOVERY_ARMS,
        ),
        "observed_only": (
            "## Recovery when there is nothing to correct (observed_only scenario)",
            CALIBRATION_ARMS,
        ),
    }
    for scenario, (title, arms) in titles.items():
        recovery = results[results["scenario"] == scenario].copy()
        if recovery.empty:
            continue
        recovery["covers"] = (recovery["lower"] <= recovery["truth"]) & (
            recovery["truth"] <= recovery["upper"]
        )
        recovery["error"] = recovery["mean"] - recovery["truth"]
        recovery["width"] = recovery["upper"] - recovery["lower"]
        recovery["interval_score"] = _interval_score(recovery)
        rows = []
        for arm in arms:
            frame = recovery[recovery["arm"] == arm]
            if frame.empty:
                continue
            low, high = _binomial_ci(int(frame["covers"].sum()), len(frame))
            rows.append(
                {
                    "arm": arm,
                    "n": len(frame),
                    "coverage": f"{frame['covers'].mean():.2f} [{low:.2f}, {high:.2f}]",
                    "mean error": f"{frame['error'].mean():.2f}",
                    "mean abs error": f"{frame['error'].abs().mean():.2f}",
                    "mean width": f"{frame['width'].mean():.2f}",
                    "interval score": f"{frame['interval_score'].mean():.2f}",
                    "divergences (median / p90)": (
                        f"{frame['divergences'].median():.0f} / "
                        f"{frame['divergences'].quantile(0.9):.0f}"
                    ),
                }
            )
        lines += [title, "", pd.DataFrame(rows).to_markdown(index=False), ""]
    lines += [
        "Coverage is of the 94% interval, with an exact 95% binomial CI. The "
        "interval score is the Winkler score (lower is better): width plus "
        "2/0.06 times the distance by which the interval misses the truth. "
        "Divergences are counted per fit, out of the fit's post-warmup draws.",
        "",
    ]

    gamma_rows = []
    for (scenario, arm), frame in results.dropna(subset=["gamma_tv_lower"]).groupby(
        ["scenario", "arm"], sort=False
    ):
        for channel in ["tv", "digital"]:
            excludes = (frame[f"gamma_{channel}_lower"] > 0) | (
                frame[f"gamma_{channel}_upper"] < 0
            )
            low, high = _binomial_ci(int(excludes.sum()), len(frame))
            gamma_rows.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "channel": channel,
                    "n": len(frame),
                    "interval excludes 0": f"{excludes.mean():.2f} [{low:.2f}, {high:.2f}]",
                    "mean P(gamma > 0)": f"{frame[f'gamma_{channel}_prob_positive'].mean():.2f}",
                }
            )
    if gamma_rows:
        lines += [
            "## Control-function coefficient",
            "",
            pd.DataFrame(gamma_rows).to_markdown(index=False),
            "",
            "In the forecast scenario TV's budget follows demand, so its interval "
            "should exclude zero (power). In the observed_only scenario spend is "
            "exogenous given the controls, so any exclusion is a false alarm.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    """Run the study, or summarise existing results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("budget_model_study"))
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.csv"
    done = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()

    if not args.summarise_only:
        finished = set()
        if not done.empty:
            finished = set(
                zip(done["scenario"], done["arm"], done["seed"], strict=True)
            )
        jobs = [
            job for job in _jobs(args.n_seeds, args.first_seed) if job not in finished
        ]
        sample_kwargs = {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "target_accept": 0.95,
            "progressbar": False,
        }
        print(f"{len(jobs)} fits to run, {len(finished)} already done.")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fit_one, *job, sample_kwargs): job for job in jobs}
            for i, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"[{i}/{len(jobs)}] {job} failed: {exc}")
                    continue
                pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
                    results_path,
                    mode="a",
                    header=not results_path.exists(),
                    index=False,
                )
                print(f"[{i}/{len(jobs)}] {job} done")
        done = pd.read_csv(results_path)

    summary = summarise(done)
    (args.out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
