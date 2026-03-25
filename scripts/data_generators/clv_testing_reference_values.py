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
"""Pre-compute reference values from the legacy lifetimes library for unit testing and save in .npy format.

Reference values are hard-coded into unit tests. This script is retained for documentation purposes.

Sources:
- BG/NBD: Fader, Hardie & Lee (2005), "Counting Your Customers the Easy Way"
  http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
- Pareto/NBD: Schmittlein, Morrison & Colombo (1987),
  "Counting Your Customers: Who Are They and What Will They Do Next?"
  https://doi.org/10.1287/mnsc.33.1.1
- MBG/NBD: Batislam, Denizel & Filiztekin (2007),
  "Empirical validation and comparison of models for customer base analysis"
  https://doi.org/10.1016/j.ijresmar.2006.12.005
- BG/BB: Fader, Hardie & Shang (2010),
"Customer-Base Analysis in a Discrete-Time Noncontractual Setting"
  https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf
"""

import os

import numpy as np
import pandas as pd
from lifetimes import (
    BetaGeoBetaBinomFitter,
    BetaGeoFitter,
    ModifiedBetaGeoFitter,
    ParetoNBDFitter,
)
from lifetimes.generate_data import beta_geometric_beta_binom_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "clv", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name, arr):
    """Save a numpy array as a .npy fixture file."""
    path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(path, arr)
    print(f"  Saved {name}.npy  shape={np.asarray(arr).shape}")


def compute_bg_nbd():
    """BG/NBD model reference values on CDNOW dataset."""
    print("\n=== BG/NBD (BetaGeoFitter) ===")
    a, b, alpha, r = 0.793, 2.426, 4.414, 0.243

    data = pd.read_csv("data/clv_quickstart.csv")
    frequency = data["frequency"]
    recency = data["recency"]
    T = data["T"]

    fitter = BetaGeoFitter()
    fitter.params_ = {"a": a, "b": b, "alpha": alpha, "r": r}

    for t in [1, 3, 6]:
        purchases = fitter.conditional_expected_number_of_purchases_up_to_time(
            t=t, frequency=frequency, recency=recency, T=T
        )
        save(f"bg_nbd_expected_purchases_t{t}", purchases)

    for t in [1, 3, 6]:
        purchases_new = fitter.expected_number_of_purchases_up_to_time(t=t)
        save(f"bg_nbd_expected_purchases_new_customer_t{t}", np.array(purchases_new))

    prob_alive = fitter.conditional_probability_alive(
        frequency=frequency, recency=recency, T=T
    )
    save("bg_nbd_probability_alive", prob_alive)


def compute_pareto_nbd():
    """Pareto/NBD model reference values on CDNOW dataset."""
    print("\n=== Pareto/NBD (ParetoNBDFitter) ===")
    r, alpha, s, beta = 0.5534, 10.5802, 0.6061, 11.6562

    data = pd.read_csv("data/clv_quickstart.csv")
    frequency = data["frequency"]
    recency = data["recency"]
    T = data["T"]

    fitter = ParetoNBDFitter()
    fitter.params_ = {"r": r, "alpha": alpha, "s": s, "beta": beta}

    for t in [1, 3, 6]:
        purchases = fitter.conditional_expected_number_of_purchases_up_to_time(
            t=t, frequency=frequency, recency=recency, T=T
        )
        save(f"pareto_nbd_expected_purchases_t{t}", purchases)

    for t in [1, 3, 6]:
        purchases_new = fitter.expected_number_of_purchases_up_to_time(t=t)
        save(
            f"pareto_nbd_expected_purchases_new_customer_t{t}",
            np.array(purchases_new),
        )

    prob_alive = fitter.conditional_probability_alive(
        frequency=frequency, recency=recency, T=T
    )
    save("pareto_nbd_probability_alive", prob_alive)

    for n_purchases, future_t in [(0, 0), (1, 1), (2, 2)]:
        prob_purchase = fitter.conditional_probability_of_n_purchases_up_to_time(
            n_purchases, future_t, frequency=frequency, recency=recency, T=T
        )
        save(
            f"pareto_nbd_purchase_probability_n{n_purchases}_t{future_t}",
            prob_purchase,
        )


def compute_mbg_nbd():
    """MBG/NBD model reference values on CDNOW dataset."""
    print("\n=== MBG/NBD (ModifiedBetaGeoFitter) ===")
    a, b, alpha, r = 0.891, 1.614, 6.183, 0.525

    data = pd.read_csv("data/clv_quickstart.csv")
    frequency = data["frequency"]
    recency = data["recency"]
    T = data["T"]

    fitter = ModifiedBetaGeoFitter()
    fitter.params_ = {"a": a, "b": b, "alpha": alpha, "r": r}

    for t in [1, 3, 6]:
        purchases = fitter.conditional_expected_number_of_purchases_up_to_time(
            t=t, frequency=frequency, recency=recency, T=T
        )
        save(f"mbg_nbd_expected_purchases_t{t}", purchases)

    for t in [1, 3, 6]:
        purchases_new = fitter.expected_number_of_purchases_up_to_time(t=t)
        save(f"mbg_nbd_expected_purchases_new_customer_t{t}", np.array(purchases_new))

    prob_alive = fitter.conditional_probability_alive(
        frequency=frequency, recency=recency, T=T
    )
    save("mbg_nbd_probability_alive", prob_alive)


def compute_bgbb():
    """BG/BB model reference values on donations dataset."""
    print("\n=== BG/BB (BetaGeoBetaBinomFitter) ===")
    alpha, beta, delta, gamma = 1.2035, 0.7497, 2.7834, 0.6567

    data = pd.read_csv("data/bgbb_donations.csv")
    test_customer_ids = [  # noqa: F841
        3463,
        4554,
        4831,
        4960,
        5038,
        5159,
        5286,
        5899,
        6154,
        6309,
        6482,
        6716,
        7038,
        7219,
        7444,
        7801,
        8041,
        8235,
        8837,
        9172,
        9900,
        11103,
    ]
    pred_data = data.query("customer_id.isin(@test_customer_ids)")

    fitter = BetaGeoBetaBinomFitter()
    fitter.params_ = {"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma}

    for t in [1, 3, 6]:
        purchases = fitter.conditional_expected_number_of_purchases_up_to_time(
            m_periods_in_future=t,
            frequency=pred_data["frequency"],
            recency=pred_data["recency"],
            n_periods=pred_data["T"],
        )
        save(f"bgbb_expected_purchases_t{t}", purchases)

    for t in [1, 3, 6]:
        prob_alive = fitter.conditional_probability_alive(
            m_periods_in_future=t,
            frequency=pred_data["frequency"],
            recency=pred_data["recency"],
            n_periods=pred_data["T"],
        )
        save(f"bgbb_probability_alive_t{t}", prob_alive)


def compute_distribution_logp_values():
    """Pre-compute logp reference values for test_distributions.py.

    These are small enough to print for direct hardcoding in tests.
    """
    print("\n=== Distribution logp reference values ===")

    print("\n--- ParetoNBD logp values ---")
    test_cases = [
        (np.array([1.5, 1]), 0.55, 10.58, 0.61, 11.67, 12),
        (np.array([1.5, 1]), [0.45, 0.55], 10.58, 0.61, 11.67, 12),
        (np.array([1.5, 1]), [0.45, 0.55], 10.58, [0.71, 0.61], 11.67, 12),
        (np.array([[1.5, 1], [5.3, 4], [6, 2]]), 0.55, 11.67, 0.61, 10.58, [12, 10, 8]),
        (np.array([1.5, 1]), 0.55, 10.58, 0.61, np.full((5, 3), 11.67), 12),
    ]

    for i, (value, r, alpha, s, beta, T) in enumerate(test_cases):
        vectorized_logp = np.vectorize(
            lambda r, alpha, s, beta, freq, rec, T: (
                ParetoNBDFitter._conditional_log_likelihood(
                    (r, alpha, s, beta), freq, rec, T
                )
            )
        )
        result = vectorized_logp(r, alpha, s, beta, value[..., 1], value[..., 0], T)
        print(f"  Case {i}: {result!r}")
        save(f"pareto_nbd_logp_case{i}", result)

    print("\n--- MBG/NBD logp values ---")
    from lifetimes import ModifiedBetaGeoFitter as MBG

    mbg_test_cases = [
        (np.array([1.5, 1]), 0.55, 10.58, 0.61, 11.67, 12),
        (np.array([1.5, 1]), [0.45, 0.55], 10.58, 0.61, 11.67, 12),
        (np.array([1.5, 1]), [0.45, 0.55], 10.58, [0.71, 0.61], 11.67, 12),
        (np.array([[1.5, 1], [5.3, 4], [6, 2]]), 0.55, 11.67, 0.61, 10.58, [12, 10, 8]),
        (np.array([1.5, 1]), 0.55, 10.58, 0.61, np.full((1), 11.67), 12),
    ]

    for i, (value, r, alpha, a, b, T) in enumerate(mbg_test_cases):
        vectorized_logp = np.vectorize(
            lambda r, alpha, a, b, freq, rec, T: (
                -1.0
                * MBG._negative_log_likelihood(
                    (np.log(r), np.log(alpha), np.log(a), np.log(b)),
                    freq,
                    rec,
                    T,
                    np.array(1),
                    0.0,
                )
            )
        )
        result = vectorized_logp(r, alpha, a, b, value[..., 1], value[..., 0], T)
        print(f"  Case {i}: {result!r}")
        save(f"mbg_nbd_logp_case{i}", result)

    print("\n--- BG/BB synthetic data for test_beta_geo_beta_binom_sample_prior ---")
    T_true = 60
    alpha_true, beta_true, gamma_true, delta_true = 1.204, 0.750, 0.657, 2.783
    np.random.seed(42)
    lt_bgbb = beta_geometric_beta_binom_model(
        N=T_true,
        alpha=alpha_true,
        beta=beta_true,
        gamma=gamma_true,
        delta=delta_true,
        size=1000,
    )
    save("bgbb_synthetic_frequency", lt_bgbb["frequency"].values)
    save("bgbb_synthetic_recency", lt_bgbb["recency"].values)
    freq_shape = lt_bgbb["frequency"].shape
    rec_shape = lt_bgbb["recency"].shape
    print(f"  Saved bgbb synthetic data: frequency {freq_shape}, recency {rec_shape}")


if __name__ == "__main__":
    compute_bg_nbd()
    compute_pareto_nbd()
    compute_mbg_nbd()
    compute_bgbb()
    compute_distribution_logp_values()
    print(f"\nAll reference values saved to {os.path.abspath(OUTPUT_DIR)}")
