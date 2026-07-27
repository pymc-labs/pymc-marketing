"""Simulate a synthetic social-media campaign world for MMM disaggregation research.

Generative story (user-level, so reach/CTR dynamics emerge mechanically):

1. A universe of ``N_USERS`` users, each belonging to one of ``K`` interest
   groups and carrying an individual responsiveness score ``r_u``.
2. Seven campaigns on one "social" channel target overlapping subsets of the
   groups -> audience sets with a known NxN overlap matrix. One of them
   (``broad_prospecting``) kicks off in April against a very large audience,
   so its cumulative reach is still growing when the window closes.
3. Every day, each active campaign buys impressions: ``spend / CPM * 1000``.
   CPM inflates with audience penetration (auction depth). Impressions land on
   the daily-online subset of the audience, weighted by responsiveness and a
   mild retargeting bias -> cumulative reach follows a C-shaped saturating
   curve with a knee, daily frequency keeps growing after the knee.
4. Clicks: per-impression click prob = base CTR x fresh-user boost x
   exp(-decay * prior frequency) -> CTR degrades once the pool saturates.
5. True incremental conversions: exposures build per-user, per-campaign
   "goodwill" (fresh exposures worth much more than repeats); goodwill decays
   daily (user-level adstock). Daily conversion hazard = sum of goodwill;
   realized conversions are attributed to campaigns by hazard share ->
   GROUND-TRUTH campaign contributions.
6. Sales y(t) = baseline trend + search-channel effect + value-per-conversion
   x social conversions + noise.
7. A pymc-marketing MMM (geometric adstock + logistic saturation) is fit on
   (date, social spend, search spend, trend, y). The posterior of
   original-scale channel contribution curves is saved -> this is "what the
   model delivers" and what the disaggregation methods must distribute.

Outputs (this folder):
- campaign_cumulative_reach.csv   dates x campaigns, cumulative unique users
- campaign_daily_metrics.csv      long: date, campaign, spend, impressions,
                                  reach (daily uniques), clicks, active flag
- audience_overlap_matrix.csv     NxN shared-audience counts (diag = size)
- mmm_input_data.csv              date, social, search, trend, y
- channel_contribution_posterior.nc  posterior draws of original-scale
                                  contribution (chain, draw, date, channel)
- ground_truth_campaign_contributions.csv  date x campaign true incremental
                                  revenue (oracle - validation only)
- ground_truth_channel_components.csv      date, baseline, search, social
- ground_truth_meta.json          audiences, params, union reach, etc.

Run:  uv run python simulate_campaign_data.py
(The branch requires pymc>=6 / arviz>=1.2 - use the repo uv env, not stale
conda envs.)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SEED = 42
N_USERS = 200_000
N_DAYS = 120
DATES = pd.date_range("2025-01-01", periods=N_DAYS, freq="D")
GROUP_PROBS = [0.35, 0.30, 0.20, 0.15]
P_ONLINE = 0.35  # fraction of audience in-auction on a given day
P_CHURN = 0.012  # daily user turnover (new users / cookie resets) -> reach tail
BURN_DAYS = 45  # warm-up so always-on campaigns start at steady state
VALUE_PER_CONVERSION = 25.0
OUT_DIR = Path(__file__).parent

CAMPAIGNS = [
    {
        "name": "brand_always_on",
        "groups": {0: 0.55, 1: 0.40},
        "start": 0,
        "end": 119,
        "budget": 320.0,
        "cpm": 7.0,
        "base_ctr": 0.014,
        "fresh_ctr_boost": 1.4,
        "ctr_freq_decay": 0.10,
        "beta_fresh": 0.00231,
        "beta_repeat": 0.00088,
        "retention": 0.80,
    },
    {
        "name": "spring_promo",
        "groups": {0: 0.60},
        "start": 10,
        "end": 75,
        "budget": 160.0,
        "cpm": 8.5,
        "base_ctr": 0.020,
        "fresh_ctr_boost": 1.8,
        "ctr_freq_decay": 0.16,
        "beta_fresh": 0.0033,
        "beta_repeat": 0.00072,
        "retention": 0.70,
    },
    {
        "name": "lookalike_conversions",
        "groups": {1: 0.35, 2: 0.55},
        "start": 0,
        "end": 119,
        "budget": 130.0,
        "cpm": 10.0,
        "base_ctr": 0.017,
        "fresh_ctr_boost": 1.2,
        "ctr_freq_decay": 0.08,
        "beta_fresh": 0.00302,
        "beta_repeat": 0.00121,
        "retention": 0.82,
    },
    {
        "name": "video_product_launch",
        "groups": {2: 0.50, 3: 0.45},
        "start": 25,
        "end": 95,
        "budget": 170.0,
        "cpm": 6.5,
        "base_ctr": 0.011,
        "fresh_ctr_boost": 2.2,
        "ctr_freq_decay": 0.20,
        "beta_fresh": 0.00209,
        "beta_repeat": 0.00049,
        "retention": 0.72,
    },
    {
        "name": "flash_sale_burst",
        "groups": {0: 0.30, 3: 0.50},
        "start": 45,
        "end": 59,  # killed after 15 days -> right-censored knee
        "budget": 300.0,
        "cpm": 9.0,
        "base_ctr": 0.024,
        "fresh_ctr_boost": 1.6,
        "ctr_freq_decay": 0.14,
        "beta_fresh": 0.00385,
        "beta_repeat": 0.00099,
        "retention": 0.60,
    },
    {
        "name": "niche_interest",
        "groups": {3: 0.60},
        "start": 0,
        "end": 119,
        "budget": 60.0,
        "cpm": 11.0,
        "base_ctr": 0.019,
        "fresh_ctr_boost": 1.3,
        "ctr_freq_decay": 0.12,
        "beta_fresh": 0.00275,
        "beta_repeat": 0.0011,
        "retention": 0.85,
    },
    {
        # April kickoff, broad prospecting: audience is huge relative to the
        # daily delivery, so penetration stays low and cumulative reach is
        # still climbing when the observation window closes (no saturation
        # knee) -> a left-censored growth curve.
        "name": "broad_prospecting",
        "groups": {0: 0.40, 1: 0.40, 2: 0.45, 3: 0.40},
        "start": 90,  # 2025-04-01
        "end": 119,
        "budget": 10.0,
        "cpm": 8.0,
        "base_ctr": 0.015,
        "fresh_ctr_boost": 1.5,
        "ctr_freq_decay": 0.12,
        "beta_fresh": 0.0026,
        "beta_repeat": 0.00095,
        "retention": 0.78,
    },
]


def build_world(rng):
    """Assign users to groups, responsiveness, and campaign audiences."""
    groups = rng.choice(len(GROUP_PROBS), size=N_USERS, p=GROUP_PROBS)
    responsiveness = rng.gamma(shape=2.0, scale=0.5, size=N_USERS)
    audiences = []
    for cfg in CAMPAIGNS:
        p_incl = np.zeros(N_USERS)
        for g, p in cfg["groups"].items():
            p_incl[groups == g] = p
        audiences.append(rng.random(N_USERS) < p_incl)
    return groups, responsiveness, np.array(audiences)  # (C, U) bool


def simulate_delivery(rng, audiences, responsiveness):
    """Daily loop: spend -> impressions -> exposure -> clicks -> conversions."""
    n_c = len(CAMPAIGNS)
    exposed_ever = np.zeros((n_c, N_USERS), dtype=bool)  # since campaign birth
    reached_window = np.zeros((n_c, N_USERS), dtype=bool)  # since obs window start
    # recency-weighted exposure count: fatigue builds with frequency but
    # recovers as people forget (decays daily)
    freq = np.zeros((n_c, N_USERS), dtype=np.float32)
    FREQ_RECOVERY = 0.85
    goodwill = np.zeros((n_c, N_USERS), dtype=np.float64)
    converted = np.zeros(N_USERS, dtype=bool)

    weekday_mult = np.array([1.0, 1.05, 1.05, 1.0, 0.95, 0.85, 0.8])
    metrics = []  # long rows
    cum_reach = np.zeros((N_DAYS, n_c), dtype=np.int64)
    true_conv = np.zeros((N_DAYS, n_c))

    # negative t = burn-in: always-on campaigns run so the window opens at
    # steady state; launches within the window keep their true fresh phases
    for t in range(-BURN_DAYS, N_DAYS):
        online = rng.random(N_USERS) < P_ONLINE

        # audience turnover: churned ids behave like brand-new users (also
        # mimics cookie resets) -> cumulative reach keeps a slow linear tail
        churned = rng.random(N_USERS) < P_CHURN
        exposed_ever[:, churned] = False
        freq[:, churned] = 0
        goodwill[:, churned] = 0.0
        converted[churned] = False
        freq *= FREQ_RECOVERY  # fatigue recovers as people forget

        dow = (DATES[0].dayofweek + t) % 7
        for c, cfg in enumerate(CAMPAIGNS):
            active = (cfg["start"] <= t <= cfg["end"]) or (t < 0 and cfg["start"] == 0)
            spend = impressions = day_reach = clicks = 0.0
            if active:
                ramp = min(1.0, (t - cfg["start"] + 1) / 4.0) if t >= 0 else 1.0
                pulse = 1.0 + 0.5 * np.sin(2 * np.pi * (t + 9 * c) / 28.0)
                spend = (
                    cfg["budget"]
                    * weekday_mult[dow]
                    * ramp
                    * pulse
                    * rng.lognormal(0.0, 0.15)
                )
                penetration = exposed_ever[c].sum() / max(audiences[c].sum(), 1)
                cpm = (
                    cfg["cpm"]
                    * (1.0 + 0.9 * penetration**1.5)
                    * rng.lognormal(0.0, 0.05)
                )
                impressions = int(spend / cpm * 1000)

                pool = np.flatnonzero(audiences[c] & online)
                if impressions > 0 and pool.size > 0:
                    w = (0.2 + responsiveness[pool]) * (
                        1.0 + 0.6 * exposed_ever[c, pool]
                    )
                    counts = rng.multinomial(impressions, w / w.sum())
                    hit = counts > 0
                    hit_idx = pool[hit]
                    hit_counts = counts[hit]

                    is_new = ~exposed_ever[c, hit_idx]
                    day_reach = int(hit.sum())

                    # clicks: fresh boost + frequency fatigue + responsiveness
                    r_norm = np.clip(responsiveness[hit_idx], 0.2, 3.0)
                    ctr_u = (
                        cfg["base_ctr"]
                        * (1.0 + cfg["fresh_ctr_boost"] * is_new)
                        * np.exp(-cfg["ctr_freq_decay"] * freq[c, hit_idx])
                        * (0.4 + 0.6 * r_norm)
                    )
                    clicks = int(rng.poisson(float((hit_counts * ctr_u).sum())))

                    # goodwill: fresh exposures >> repeats, fatigue on repeats
                    repeat_counts = np.where(is_new, hit_counts - 1, hit_counts)
                    add = cfg["beta_fresh"] * is_new + cfg["beta_repeat"] * np.minimum(
                        repeat_counts, 6
                    ) * np.exp(-0.12 * freq[c, hit_idx]) * (0.4 + 0.6 * r_norm)
                    goodwill[c, hit_idx] += add
                    exposed_ever[c, hit_idx] = True
                    if t >= 0:
                        # platform cumulative-reach semantics: unique IDs
                        # counted within the query window are never un-counted
                        reached_window[c, hit_idx] = True
                    freq[c, hit_idx] += hit_counts

            if t >= 0:
                metrics.append(
                    {
                        "date": DATES[t],
                        "campaign": cfg["name"],
                        "active": bool(active),
                        "spend": round(float(spend), 2),
                        "impressions": int(impressions),
                        "reach": int(day_reach),
                        "clicks": int(clicks),
                    }
                )
                cum_reach[t, c] = int(reached_window[c].sum())

        # conversions from total goodwill (competing campaigns share credit)
        hazard = goodwill.sum(axis=0)
        eligible = (~converted) & (hazard > 1e-12)
        p_conv = 1.0 - np.exp(-hazard[eligible])
        conv_mask = rng.random(eligible.sum()) < p_conv
        conv_idx = np.flatnonzero(eligible)[conv_mask]
        if conv_idx.size:
            if t >= 0:
                shares = goodwill[:, conv_idx] / hazard[conv_idx]
                true_conv[t] = shares.sum(axis=1)
            converted[conv_idx] = True
        goodwill *= np.array([cfg["retention"] for cfg in CAMPAIGNS])[:, None]

    return pd.DataFrame(metrics), cum_reach, true_conv, exposed_ever


def geometric_adstock(x, alpha, l_max=8):
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    return np.convolve(x, w)[: len(x)]


def make_target(rng, true_conv):
    """Baseline + closed-form search channel + social conversions + noise."""
    trend = np.linspace(0.0, 1.0, N_DAYS)
    baseline = 7000.0 + 900.0 * trend
    search_spend = 260.0 * rng.lognormal(0.0, 0.18, size=N_DAYS)
    x = geometric_adstock(search_spend / search_spend.max(), alpha=0.45)
    lam = 4.0
    search_effect = 1400.0 * (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))
    social_effect = VALUE_PER_CONVERSION * true_conv.sum(axis=1)
    noise = rng.normal(0.0, 130.0, size=N_DAYS)
    y = baseline + search_effect + social_effect + noise
    return trend, baseline, search_spend, search_effect, social_effect, y


def fit_mmm(mmm_df):
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    mmm = MMM(
        date_column="date",
        channel_columns=["social", "search"],
        control_columns=["trend"],
        adstock=GeometricAdstock(l_max=12),
        saturation=LogisticSaturation(),
    )
    X = mmm_df[["date", "social", "search", "trend"]].copy()
    y = mmm_df["y"]
    mmm.build_model(X, y)
    mmm.add_original_scale_contribution_variable(var=["channel_contribution"])
    mmm.fit(
        X,
        y,
        chains=4,
        draws=800,
        tune=2000,
        target_accept=0.98,
        random_seed=SEED,
        progressbar=False,
    )
    return mmm


def main():
    rng = np.random.default_rng(SEED)
    names = [cfg["name"] for cfg in CAMPAIGNS]

    _groups, responsiveness, audiences = build_world(rng)
    overlap = (audiences.astype(np.int64) @ audiences.T.astype(np.int64)).astype(int)
    print("audience sizes:", dict(zip(names, np.diag(overlap), strict=True)))

    metrics_df, cum_reach, true_conv, exposed_ever = simulate_delivery(
        rng, audiences, responsiveness
    )
    trend, baseline, search_spend, search_effect, social_effect, y = make_target(
        rng, true_conv
    )

    social_spend = (
        metrics_df.pivot(index="date", columns="campaign", values="spend")
        .loc[DATES, names]
        .sum(axis=1)
        .to_numpy()
    )
    mmm_df = pd.DataFrame(
        {
            "date": DATES,
            "social": social_spend,
            "search": search_spend,
            "trend": trend,
            "y": y,
        }
    )

    # --- save platform-style datasets -------------------------------------
    pd.DataFrame(cum_reach, index=DATES, columns=names).rename_axis("date").to_csv(
        OUT_DIR / "campaign_cumulative_reach.csv"
    )
    metrics_df.to_csv(OUT_DIR / "campaign_daily_metrics.csv", index=False)
    pd.DataFrame(overlap, index=names, columns=names).rename_axis("campaign").to_csv(
        OUT_DIR / "audience_overlap_matrix.csv"
    )
    mmm_df.to_csv(OUT_DIR / "mmm_input_data.csv", index=False)

    # --- oracle files ------------------------------------------------------
    truth = pd.DataFrame(
        VALUE_PER_CONVERSION * true_conv, index=DATES, columns=names
    ).rename_axis("date")
    truth.to_csv(OUT_DIR / "ground_truth_campaign_contributions.csv")
    pd.DataFrame(
        {
            "date": DATES,
            "baseline": baseline,
            "search_effect": search_effect,
            "social_effect": social_effect,
        }
    ).to_csv(OUT_DIR / "ground_truth_channel_components.csv", index=False)

    union_reached = int((exposed_ever.any(axis=0)).sum())
    meta = {
        "seed": SEED,
        "n_users": N_USERS,
        "n_days": N_DAYS,
        "p_online_daily": P_ONLINE,
        "value_per_conversion": VALUE_PER_CONVERSION,
        "group_probs": GROUP_PROBS,
        "campaigns": CAMPAIGNS,
        "audience_sizes": {
            n: int(s) for n, s in zip(names, np.diag(overlap), strict=True)
        },
        "audience_union_size": int(audiences.any(axis=0).sum()),
        "union_users_actually_reached": union_reached,
        "true_total_contribution_by_campaign": {
            n: round(float(v), 2) for n, v in truth.sum(axis=0).items()
        },
        "notes": (
            "Ground-truth conversion credit uses hazard shares (goodwill_c / "
            "total goodwill) among converting users - the causal responsibility "
            "under the competing-exposures generative model."
        ),
    }
    (OUT_DIR / "ground_truth_meta.json").write_text(json.dumps(meta, indent=2))

    # --- fit MMM and save posterior contribution curves --------------------
    print("fitting MMM ...")
    mmm = fit_mmm(mmm_df)
    post = mmm.idata.posterior["channel_contribution_original_scale"]
    post = post.transpose("chain", "draw", "date", "channel")
    thin = max(1, post.sizes["draw"] // 100)  # ~400 total draws
    post_thin = post.isel(draw=slice(0, None, thin)).astype("float32")
    xr.Dataset({"channel_contribution": post_thin}).to_netcdf(
        OUT_DIR / "channel_contribution_posterior.nc"
    )

    social_mean = post_thin.sel(channel="social").mean(("chain", "draw")).to_numpy()
    corr = np.corrcoef(social_mean, social_effect)[0, 1]
    print(f"MMM social contribution vs truth: corr={corr:.3f}")
    print(
        f"totals - true social: {social_effect.sum():,.0f} | "
        f"MMM posterior mean: {social_mean.sum():,.0f}"
    )
    print("true campaign totals:", meta["true_total_contribution_by_campaign"])
    print("saved all outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
