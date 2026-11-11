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
   interval exclude zero (the false-alarm rate)?

The fitted models use logistic saturation by default, although the simulator
draws sales from a Michaelis-Menten curve (see the note after the results).

Usage::

    python scripts/budget_model/simulation_study.py --n-seeds 50 --workers 3 --out study
    python scripts/budget_model/simulation_study.py --summarise-only --out study

Pass ``--saturation michaelis_menten`` to fit the simulator's own curve
family; the ``observed_only`` scenario then adds a
``budget_design_misspecified`` arm fitted with logistic saturation.

Each fit is appended to ``<out>/results.csv`` as it finishes, so an interrupted
run resumes where it stopped. ``<out>/summary.md`` holds the tables.

Expected results
----------------
With the defaults (seeds 1000-1049; logistic saturation; 1,000 draws, 1,500
tuning steps and 4 chains per fit; nutpie), the study takes about two hours on
3 workers and gives the tables below. The true marginal return is about 2.
Coverage is of the 94% interval, with an exact 95% binomial CI. The interval
score is the Winkler score (width plus 2/0.06 times the distance by which the
interval misses the truth; lower is better). Divergences are per fit, out of
4,000 draws; MR R-hat is the R-hat of TV's marginal return. Every arm
completed all 50 fits; the script reports ``n`` per arm and lists any fit that
fails.

Recovery of TV's marginal return when budgets chase demand (``"forecast"``)::

    arm                 coverage           mean err  abs err  width  score  div (median/p90)  MR R-hat (median/max)
    plain               0.04 [0.00, 0.14]   +2.31     2.31    2.03   45.7   0 / 120           1.004 / 1.257
    lift_history        0.22 [0.12, 0.36]   +1.49     1.49    1.84   22.5   0 / 36            1.003 / 1.372
    lift_full           0.20 [0.10, 0.34]   +1.42     1.42    1.65   22.9   0 / 8             1.004 / 1.076
    budget_no_design    0.96 [0.86, 1.00]   +1.59     1.67    8.37    9.0   10 / 67           1.029 / 1.620
    budget_design       0.74 [0.60, 0.85]   +0.75     0.98    3.06    6.3   9 / 44            1.011 / 1.101
    budget_design_lag1  0.94 [0.83, 0.99]   +0.35     0.72    3.02    3.8   10 / 42           1.010 / 1.256
    budget_design_lag3  0.94 [0.83, 0.99]   +0.27     0.66    3.09    3.7   10 / 64           1.010 / 1.474

Recovery when spend is exogenous given the controls (``"observed_only"``)::

    arm                 coverage           mean err  abs err  width  score  div (median/p90)  MR R-hat (median/max)
    plain               0.90 [0.78, 0.97]   +0.06     0.64    2.50    3.1   2 / 11            1.006 / 1.098
    budget_design       0.92 [0.81, 0.98]   +0.24     0.86    3.21    3.9   11 / 611          1.018 / 1.477
    budget_design_lag1  0.90 [0.78, 0.97]   +0.29     0.91    3.26    4.6   10 / 54           1.013 / 1.556
    budget_design_lag3  0.86 [0.73, 0.94]   +0.33     0.93    3.30    5.3   9 / 40            1.015 / 1.296
    budget_no_design    0.98 [0.89, 1.00]   +0.97     1.22    8.18    8.3   8 / 33            1.021 / 1.383

Share of markets where TV's 94% ``gamma`` interval excludes zero (power in
``"forecast"``, false alarms in ``"observed_only"``). "Lag 0" is the
contemporaneous coefficient alone, with a nominal false-alarm rate of 6%. "Any
lag" asks whether any of the ``1 + surprise_lags`` intervals excludes zero; by
chance alone that happens about ``1 - 0.94 ** (1 + surprise_lags)`` of the time
(12% with one lag, 22% with three)::

    scenario       arm                 lag 0               any lag             mean P(gamma > 0)
    forecast       budget_design       0.64 [0.49, 0.77]   -                   0.94
    forecast       budget_design_lag1  0.66 [0.51, 0.79]   0.80 [0.66, 0.90]   0.94
    forecast       budget_design_lag3  0.66 [0.51, 0.79]   0.84 [0.71, 0.93]   0.94
    forecast       budget_no_design    0.00 [0.00, 0.07]   -                   0.71
    observed_only  budget_design       0.08 [0.02, 0.19]   -                   0.47
    observed_only  budget_design_lag1  0.08 [0.02, 0.19]   0.18 [0.09, 0.31]   0.47
    observed_only  budget_design_lag3  0.12 [0.05, 0.24]   0.40 [0.26, 0.55]   0.46
    observed_only  budget_no_design    0.06 [0.01, 0.17]   -                   0.47

Reading: every arm overstates TV when budgets chase demand. The lift
likelihood removes about a third of the plain MMM's bias, and counting the
experiment twice does not help. The budget model with the design and one
lagged surprise cuts the bias to about +0.35, covers the truth in 94% of
markets, its nominal rate, and has a sixth of the lift likelihood's interval
score; without the design, its coverage comes only from intervals almost three
times as wide. When there is nothing to correct, the same configuration gives
up about 0.2 in bias (interval score 3.1 to 4.6) while covering as often as the
plain MMM, so with a design the correction is cheap insurance, and without one
it is a poor bet. With a design, the contemporaneous ``gamma`` detects
demand-chasing budgets in about two thirds of markets, with a false-alarm rate
close to nominal; without one it never does. Reading the lagged coefficients
too raises power to about 0.80 with one lag, but the false-alarm rate rises
with it, to 0.18 against about 0.12 expected by chance (Digital, never tested,
also reaches 0.12), and to 0.40 with three lags. An interval that excludes zero
is evidence worth following up, not a verdict, and three lags make the
diagnostic unreliable: one more reason to keep ``surprise_lags`` small.

Logistic saturation misspecifies the simulator's Michaelis-Menten curve on
purpose. Every comparison reads the curve's slope near typical spend, where the
two families agree closely, and the logistic fits mix far better. With
``--saturation michaelis_menten`` (2 chains of 500 draws), the budget model's
marginal return mixes slowly along a ridge between the curve's two parameters,
and the recommended configuration covers the truth in only 82% [69%, 91%] of
markets, with a bias of +0.51. Mixing is improved, not perfect, with logistic
saturation: TV's marginal return has R-hat above 1.05 in 1 to 4 of the 50
design-arm fits per scenario when budgets chase demand, 4 to 12 when they do
not, and 8 to 12 without a design.
"""

from __future__ import annotations

import argparse
import csv
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
    "mr_rhat",
    *[
        f"gamma_{channel}_{stat}"
        for channel in ("tv", "digital")
        for stat in ("lower", "upper", "prob_positive", "any_lag_excludes")
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


def _fit_one(
    scenario: str, arm: str, seed: int, sample_kwargs: dict, saturation: str
) -> dict:
    warnings.filterwarnings("ignore")
    from pymc_marketing.mmm import BudgetModelEffect
    from pymc_marketing.mmm.synthetic_data import simulate_endogenous_spend_market

    market = simulate_endogenous_spend_market(scenario, random_seed=seed)
    data, n_hist = market.data, market.n_history
    X, y = data.drop(columns="y"), data["y"]
    effect = None
    mmm = _make_mmm("logistic" if arm.endswith("misspecified") else saturation)

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
        "mr_rhat": _rhat(values, mmm.idata.posterior.sizes["chain"]),
    }
    if effect is not None:
        summary = effect.exogeneity_summary(mmm, interval_prob=INTERVAL_PROB)
        for channel, frame in summary.groupby("channel"):
            current = frame[frame["lag"] == 0].iloc[0]
            row[f"gamma_{channel}_lower"] = float(current["gamma_lower"])
            row[f"gamma_{channel}_upper"] = float(current["gamma_upper"])
            row[f"gamma_{channel}_prob_positive"] = float(current["prob_positive"])
            row[f"gamma_{channel}_any_lag_excludes"] = bool(
                ((frame["gamma_lower"] > 0) | (frame["gamma_upper"] < 0)).any()
            )
    return row


def _rhat(values: np.ndarray, n_chains: int) -> float:
    """Rank-normalised R-hat of posterior draws stored chain by chain."""
    import arviz as az
    import xarray as xr

    draws = xr.DataArray(values.reshape(n_chains, -1), dims=("chain", "draw"))
    return float(az.rhat(draws.rename("value").to_dataset())["value"])


def _jobs(n_seeds: int, first_seed: int, saturation: str) -> list[tuple[str, str, int]]:
    seeds = range(first_seed, first_seed + n_seeds)
    # With logistic saturation everywhere, the misspecified arm is budget_design.
    calibration = [
        arm
        for arm in CALIBRATION_ARMS
        if not (saturation == "logistic" and arm.endswith("misspecified"))
    ]
    return [("forecast", arm, s) for s in seeds for arm in RECOVERY_ARMS] + [
        ("observed_only", arm, s) for s in seeds for arm in calibration
    ]


def _read_results(path: Path) -> pd.DataFrame:
    """Read results, including rows appended after ``COLUMNS`` gained fields.

    Rows are matched to the file's header by length; longer rows were written
    with the current ``COLUMNS``. The file is rewritten with ``COLUMNS`` so later
    appends line up.
    """
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        records = [
            dict(zip(header if len(row) == len(header) else COLUMNS, row, strict=True))
            for row in reader
            if row
        ]
    frame = pd.DataFrame.from_records(records).reindex(columns=COLUMNS)
    frame = frame.replace("", np.nan)
    flags = [column for column in COLUMNS if column.endswith("_any_lag_excludes")]
    for column in COLUMNS:
        if column in flags:
            frame[column] = frame[column].map({"True": True, "False": False})
        elif column not in ("scenario", "arm"):
            frame[column] = pd.to_numeric(frame[column])
    if header != COLUMNS:
        frame.to_csv(path, index=False)
    return frame


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
                    "MR R-hat (median / max)": (
                        f"{frame['mr_rhat'].median():.3f} / "
                        f"{frame['mr_rhat'].max():.3f}"
                    ),
                }
            )
        lines += [title, "", pd.DataFrame(rows).to_markdown(index=False), ""]
    lines += [
        "Coverage is of the 94% interval, with an exact 95% binomial CI. The "
        "interval score is the Winkler score (lower is better): width plus "
        "2/0.06 times the distance by which the interval misses the truth. "
        "Divergences are counted per fit, out of the fit's post-warmup draws. "
        "MR R-hat is the rank-normalised R-hat of TV's marginal return.",
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
            any_lag = frame.get(f"gamma_{channel}_any_lag_excludes")
            any_lag_text = "n/a"
            if any_lag is not None and any_lag.notna().all():
                n_any = int(any_lag.astype(bool).sum())
                any_low, any_high = _binomial_ci(n_any, len(frame))
                any_lag_text = (
                    f"{n_any / len(frame):.2f} [{any_low:.2f}, {any_high:.2f}]"
                )
            gamma_rows.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "channel": channel,
                    "n": len(frame),
                    "lag-0 interval excludes 0": (
                        f"{excludes.mean():.2f} [{low:.2f}, {high:.2f}]"
                    ),
                    "any lag excludes 0": any_lag_text,
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
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--saturation",
        choices=["michaelis_menten", "logistic"],
        default="logistic",
        help="Saturation family of every fitted MMM; the simulator always uses "
        "Michaelis-Menten.",
    )
    parser.add_argument("--out", type=Path, default=Path("budget_model_study"))
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.csv"
    done = _read_results(results_path)

    if not args.summarise_only:
        finished = set()
        if not done.empty:
            finished = set(
                zip(done["scenario"], done["arm"], done["seed"], strict=True)
            )
        jobs = [
            job
            for job in _jobs(args.n_seeds, args.first_seed, args.saturation)
            if job not in finished
        ]
        sample_kwargs = {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "target_accept": 0.95,
            "progressbar": False,
        }
        print(f"{len(jobs)} fits to run, {len(finished)} already done.")
        failed = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_fit_one, *job, sample_kwargs, args.saturation): job
                for job in jobs
            }
            for i, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"[{i}/{len(jobs)}] {job} failed: {exc}")
                    failed.append(job)
                    continue
                pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
                    results_path,
                    mode="a",
                    header=not results_path.exists(),
                    index=False,
                )
                print(f"[{i}/{len(jobs)}] {job} done")
        done = _read_results(results_path)
        if failed:
            print(
                f"{len(failed)} fits failed and are missing from the tables: {failed}. "
                "Rerun the script to retry them."
            )

    summary = summarise(done)
    (args.out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
