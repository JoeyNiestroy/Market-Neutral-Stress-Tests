"""
Market-Neutral MVO Stress Test

"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, asdict

import cvxpy as cp
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = "asset_data.duckdb"
TOP_N = 100
WINDOW = 126
ROLLING_BETA_WINDOW = 63

LAM = 1.0
ETA = 1e-3
POS_CAP = 0.05
TC_BPS = 4

RANDOM_SEED = 42

# Noise grid. Daily log returns are usually around 0.5%-3% volatility, so these
# are intentionally realistic-ish.
NOISE_DISTS = ["normal", "student_t", "mixture", "laplace"]
NOISE_MEANS = [0.0, 0.0005, -0.0005]
NOISE_STDS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
SIGNAL_SCALES = [1.0, 0.5, 0.25, 0.0, -0.25]

# to avoid running full grid
MAX_SCENARIOS: int | None = 40

SUMMARY_CSV = "noise_stress_summary.csv"
DAILY_DIAG_CSV = "noise_stress_daily_diagnostics.csv"
CUM_PNL_PLOT = "noise_stress_cumulative_pnl.png"
NORMAL_SHARPE_HEATMAP = "noise_stress_sharpe_heatmap_normal.png"
NORMAL_BETA_HEATMAP = "noise_stress_realized_beta_heatmap_normal.png"


# Data class
@dataclass(frozen=True)
class NoiseScenario:
    scenario_id: int
    dist: str
    noise_mean: float
    noise_std: float
    signal_scale: float
    seed: int

    @property
    def label(self) -> str:
        return (
            f"id={self.scenario_id} | {self.dist} | "
            f"mean={self.noise_mean:g} | std={self.noise_std:g} | "
            f"scale={self.signal_scale:g}"
        )



#Load data


def load_data(db_path: str, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    """Pull top N stocks by market cap and return wide adjusted-close DataFrame."""
    if not os.path.exists(db_path):
        sys.exit(f"[ERROR] Database not found: {db_path}")

    con = duckdb.connect(db_path, read_only=True)

    symbols_df = con.execute(
        f"""
        SELECT symbol
        FROM stock_metadata
        WHERE market_cap IS NOT NULL
        ORDER BY market_cap DESC
        LIMIT {top_n}
        """
    ).df()

    symbols = symbols_df["symbol"].tolist()

    if len(symbols) < top_n:
        print(f"[WARN] Only {len(symbols)} symbols found with market cap data.")

    sym_list = ", ".join(f"'{s}'" for s in symbols)

    prices = con.execute(
        f"""
        SELECT symbol, date, adj_close
        FROM daily_prices
        WHERE symbol IN ({sym_list})
        ORDER BY date, symbol
        """
    ).df()

    con.close()

    wide = prices.pivot(index="date", columns="symbol", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    wide = wide.dropna(axis=1, thresh=WINDOW + 50)

    symbols = wide.columns.tolist()
    print(f"[INFO] Universe: {len(symbols)} stocks after coverage filter.")

    return wide, symbols


#computing log returns, equal weighting bc lazy

def compute_returns(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Log returns and equal-weighted universe market return."""
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)
    mkt.name = "equal_weight_market_return"
    return rets, mkt




def rolling_ols(
    rets: pd.DataFrame,
    mkt: pd.Series,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each day t, estimate alpha/beta using [t-window, t-1] and compute
    out-of-sample idiosyncratic return on day t.

    Returns:
        alphas_df : alpha estimates available at t
        betas_df  : beta estimates available at t
        resid_df  : eps_hat_t = r_t - alpha_hat_t - beta_hat_t mkt_t
    """
    dates = rets.index
    stocks = rets.columns
    T, N = len(dates), len(stocks)

    alphas_arr = np.full((T, N), np.nan)
    betas_arr = np.full((T, N), np.nan)
    resid_arr = np.full((T, N), np.nan)

    mkt_vals = mkt.values
    ret_vals = rets.values

    for t in tqdm(range(window, T), desc="Rolling OLS", unit="day"):
        r_window = ret_vals[t - window : t]
        m_window = mkt_vals[t - window : t]
        X = np.column_stack([np.ones(window), m_window])

        valid = ~np.any(np.isnan(r_window), axis=0)
        if valid.sum() < 2:
            continue

        coef, _, _, _ = np.linalg.lstsq(X, r_window[:, valid], rcond=None)
        alpha_t = coef[0]
        beta_t = coef[1]

        alphas_arr[t, valid] = alpha_t
        betas_arr[t, valid] = beta_t
        resid_arr[t, valid] = ret_vals[t, valid] - alpha_t - beta_t * mkt_vals[t]

    alphas_df = pd.DataFrame(alphas_arr, index=dates, columns=stocks)
    betas_df = pd.DataFrame(betas_arr, index=dates, columns=stocks)
    resid_df = pd.DataFrame(resid_arr, index=dates, columns=stocks)

    return alphas_df, betas_df, resid_df


def draw_noise(
    rng: np.random.Generator,
    dist: str,
    mean: float,
    std: float,
    size: int,
) -> np.ndarray:
    """
    Draw forecast-error noise with target approximate mean/std.

    All distributions are centered/scaled to make std comparable across cases.
    """
    if std == 0.0:
        return np.full(size, mean)

    if dist == "normal":
        z = rng.normal(0.0, 1.0, size=size)

    elif dist == "student_t":
        df = 3
        z = rng.standard_t(df=df, size=size)
        z = z / np.sqrt(df / (df - 2))  # unit variance for df > 2

    elif dist == "laplace":
        z = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=size)  # unit variance

    elif dist == "mixture":
        # Mostly normal noise, sometimes 5-sigma shock. Standardize by the
        # theoretical mixture variance.
        shock_prob = 0.05
        is_shock = rng.uniform(size=size) < shock_prob
        z = rng.normal(0.0, 1.0, size=size)
        z[is_shock] = rng.normal(0.0, 5.0, size=is_shock.sum())
        mix_var = (1.0 - shock_prob) * 1.0**2 + shock_prob * 5.0**2
        z = z / np.sqrt(mix_var)

    else:
        raise ValueError(f"Unknown noise distribution: {dist}")

    return mean + std * z


def build_scenarios() -> list[NoiseScenario]:
    scenarios: list[NoiseScenario] = []
    sid = 0

    for dist in NOISE_DISTS:
        for signal_scale in SIGNAL_SCALES:
            for noise_mean in NOISE_MEANS:
                for noise_std in NOISE_STDS:
                    scenarios.append(
                        NoiseScenario(
                            scenario_id=sid,
                            dist=dist,
                            noise_mean=noise_mean,
                            noise_std=noise_std,
                            signal_scale=signal_scale,
                            seed=RANDOM_SEED + sid,
                        )
                    )
                    sid += 1

    if MAX_SCENARIOS is not None:
        scenarios = scenarios[:MAX_SCENARIOS]

    return scenarios


# portfolio optimization stuff, kept very basic 
def solve_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    beta: np.ndarray,
    lam: float,
    pos_cap: float,
    w_prev: np.ndarray,
    eta: float,
) -> np.ndarray | None:
    """
    Solve:
        max_w mu'w - lam w'Sigma w - eta ||w - w_prev||_1
        s.t.  beta'w = 0, sum(w)=0, |w_i| <= pos_cap
    """
    N = len(mu)
    w = cp.Variable(N)

    objective = cp.Maximize(
        mu @ w
        - lam * cp.quad_form(w, Sigma)
        - eta * cp.norm1(w - w_prev)
    )

    constraints = [
        beta @ w == 0.0,
        cp.sum(w) == 0.0,
        cp.abs(w) <= pos_cap,
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    return np.asarray(w.value).reshape(-1)



def run_scenario(
    scenario: NoiseScenario,
    rets: pd.DataFrame,
    mkt: pd.Series,
    betas_df: pd.DataFrame,
    resid_df: pd.DataFrame,
    window: int,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Run a single noise scenario."""
    dates = rets.index
    T, N = rets.shape
    rng = np.random.default_rng(scenario.seed)

    pnl = np.full(T, np.nan)
    gross_pnl = np.full(T, np.nan)
    tc_arr = np.full(T, np.nan)
    turnover_arr = np.full(T, np.nan)
    ex_ante_beta_arr = np.full(T, np.nan)
    next_day_beta_arr = np.full(T, np.nan)
    beta_drift_arr = np.full(T, np.nan)
    signal_ic_arr = np.full(T, np.nan)
    signal_rank_ic_arr = np.full(T, np.nan)
    n_valid_arr = np.full(T, np.nan)

    weights = np.zeros((T, N))
    w_prev = np.zeros(N)

    ret_vals = rets.values

    for t in range(window, T - 1):
        # Clean oracle idio signal for next day.
        clean_signal = resid_df.iloc[t + 1].values
        beta_t = betas_df.iloc[t].values
        beta_next = betas_df.iloc[t + 1].values
        r_window = rets.iloc[t - window : t].values
        r_next_full = ret_vals[t + 1]

        valid = (
            ~np.isnan(clean_signal)
            & ~np.isnan(beta_t)
            & ~np.any(np.isnan(r_window), axis=0)
            & ~np.isnan(r_next_full)
        )

        if valid.sum() < 10:
            continue

        clean_v = clean_signal[valid]
        beta_v = beta_t[valid]
        R_v = r_window[:, valid]
        nv = valid.sum()

        noise_v = draw_noise(
            rng=rng,
            dist=scenario.dist,
            mean=scenario.noise_mean,
            std=scenario.noise_std,
            size=nv,
        )

        mu_v = scenario.signal_scale * clean_v + noise_v

        # IC diagnostics against actual next-day idio return.
        if np.std(mu_v) > 0 and np.std(clean_v) > 0:
            signal_ic_arr[t] = np.corrcoef(mu_v, clean_v)[0, 1]
            signal_rank_ic_arr[t] = stats.spearmanr(mu_v, clean_v).correlation

        Sigma_v = np.cov(R_v, rowvar=False) + 1e-6 * np.eye(nv)

        w_prev_v = w_prev[valid]
        w_v = solve_portfolio(
            mu=mu_v,
            Sigma=Sigma_v,
            beta=beta_v,
            lam=LAM,
            pos_cap=POS_CAP,
            w_prev=w_prev_v,
            eta=ETA,
        )

        if w_v is None:
            continue

        w_full = np.zeros(N)
        w_full[valid] = w_v
        weights[t] = w_full

        turnover = np.sum(np.abs(w_full - w_prev))
        tc = TC_BPS * 1e-4 * turnover
        gross = float(w_full @ r_next_full)

        pnl[t] = gross - tc
        gross_pnl[t] = gross
        tc_arr[t] = tc
        turnover_arr[t] = turnover
        ex_ante_beta_arr[t] = float(np.nansum(w_full * beta_t))
        next_day_beta_arr[t] = float(np.nansum(w_full * beta_next))
        beta_drift_arr[t] = next_day_beta_arr[t] - ex_ante_beta_arr[t]
        n_valid_arr[t] = nv

        w_prev = w_full.copy()

    pnl_series = pd.Series(pnl, index=dates, name=str(scenario.scenario_id)).dropna()

    diagnostics = pd.DataFrame(
        {
            "scenario_id": scenario.scenario_id,
            "date": dates,
            "gross_pnl": gross_pnl,
            "pnl": pnl,
            "tc": tc_arr,
            "turnover": turnover_arr,
            "ex_ante_beta": ex_ante_beta_arr,
            "next_day_beta": next_day_beta_arr,
            "beta_drift": beta_drift_arr,
            "abs_beta_drift": np.abs(beta_drift_arr),
            "signal_ic": signal_ic_arr,
            "signal_rank_ic": signal_rank_ic_arr,
            "n_valid": n_valid_arr,
        }
    ).dropna(subset=["pnl"])

    return pnl_series, diagnostics, weights


# Metrics related functions


def compute_realized_beta(pnl_series: pd.Series, mkt: pd.Series) -> float:
    aligned = pd.concat([pnl_series.rename("pnl"), mkt.rename("mkt")], axis=1).dropna()
    if len(aligned) < 10 or aligned["mkt"].var() == 0:
        return np.nan
    return float(aligned["pnl"].cov(aligned["mkt"]) / aligned["mkt"].var())


def compute_rolling_market_beta(
    pnl_series: pd.Series,
    mkt: pd.Series,
    window: int,
) -> pd.Series:
    aligned = pd.concat([pnl_series.rename("pnl"), mkt.rename("mkt")], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return (aligned["pnl"].rolling(window).cov(aligned["mkt"]) / aligned["mkt"].rolling(window).var()).dropna()


def compute_summary(
    scenario: NoiseScenario,
    pnl_series: pd.Series,
    diagnostics: pd.DataFrame,
    mkt: pd.Series,
) -> dict:
    ann = 252
    daily_mean = pnl_series.mean()
    daily_std = pnl_series.std()
    sharpe = daily_mean / daily_std * np.sqrt(ann) if daily_std > 0 else np.nan
    ann_ret = daily_mean * ann
    ann_vol = daily_std * np.sqrt(ann)

    cum = (1.0 + pnl_series).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()

    rb = compute_rolling_market_beta(pnl_series, mkt, ROLLING_BETA_WINDOW)

    row = asdict(scenario)
    row.update(
        {
            "label": scenario.label,
            "n_days": len(pnl_series),
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": dd.min(),
            "realized_beta": compute_realized_beta(pnl_series, mkt),
            "mean_abs_rolling_beta": rb.abs().mean() if len(rb) else np.nan,
            "p95_abs_rolling_beta": rb.abs().quantile(0.95) if len(rb) else np.nan,
            "max_abs_rolling_beta": rb.abs().max() if len(rb) else np.nan,
            "mean_turnover": diagnostics["turnover"].mean(),
            "p95_turnover": diagnostics["turnover"].quantile(0.95),
            "mean_abs_beta_drift": diagnostics["abs_beta_drift"].mean(),
            "p95_abs_beta_drift": diagnostics["abs_beta_drift"].quantile(0.95),
            "mean_signal_ic": diagnostics["signal_ic"].mean(),
            "mean_signal_rank_ic": diagnostics["signal_rank_ic"].mean(),
            "mean_tc": diagnostics["tc"].mean(),
        }
    )
    return row

#plotting junk

def plot_top_cumulative_pnl(pnl_by_scenario: dict[int, pd.Series], summary: pd.DataFrame) -> None:
    """Plot baseline, worst, and best scenarios for quick visual inspection."""
    if summary.empty:
        return

    chosen_ids = set()

    # baseline-ish scenario: normal, zero mean, zero noise, scale=1.
    base = summary[
        (summary["dist"] == "normal")
        & (summary["noise_mean"] == 0.0)
        & (summary["noise_std"] == 0.0)
        & (summary["signal_scale"] == 1.0)
    ]
    if not base.empty:
        chosen_ids.add(int(base.iloc[0]["scenario_id"]))

    chosen_ids.update(summary.nlargest(3, "sharpe")["scenario_id"].astype(int).tolist())
    chosen_ids.update(summary.nsmallest(3, "sharpe")["scenario_id"].astype(int).tolist())
    chosen_ids.update(summary.nlargest(3, "max_abs_rolling_beta")["scenario_id"].astype(int).tolist())

    plt.figure(figsize=(15, 7))
    for sid in sorted(chosen_ids):
        s = pnl_by_scenario[sid]
        label = summary.loc[summary["scenario_id"] == sid, "label"].iloc[0]
        cum = (1.0 + s).cumprod()
        plt.plot(cum.index, cum.values, linewidth=1.2, label=label)

    plt.title("Cumulative PnL: Baseline, Best/Worst Sharpe, Worst Beta Leakage")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CUM_PNL_PLOT, dpi=150)
    print(f"[PLOT] Saved {CUM_PNL_PLOT}")


def plot_normal_heatmaps(summary: pd.DataFrame) -> None:
    """Heatmaps for the normal/zero-mean slice."""
    df = summary[(summary["dist"] == "normal") & (summary["noise_mean"] == 0.0)].copy()
    if df.empty:
        return

    for signal_scale in sorted(df["signal_scale"].unique(), reverse=True):
        sub = df[df["signal_scale"] == signal_scale]
        if sub.empty:
            continue

        # Sharpe by noise std. Since mean is fixed, this is a one-dimensional
        # grid, but plot as a simple line for readability.
        sub = sub.sort_values("noise_std")

        plt.figure(figsize=(9, 5))
        plt.plot(sub["noise_std"], sub["sharpe"], marker="o")
        plt.axhline(0.0, linestyle="--", linewidth=0.8)
        plt.title(f"Sharpe vs Noise Std | normal mean=0 scale={signal_scale:g}")
        plt.xlabel("Noise std")
        plt.ylabel("Annualized Sharpe")
        plt.grid(alpha=0.3)
        out = NORMAL_SHARPE_HEATMAP.replace(".png", f"_scale_{signal_scale:g}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"[PLOT] Saved {out}")

        plt.figure(figsize=(9, 5))
        plt.plot(sub["noise_std"], sub["max_abs_rolling_beta"], marker="o")
        plt.title(f"Worst Rolling Market Beta vs Noise Std | normal mean=0 scale={signal_scale:g}")
        plt.xlabel("Noise std")
        plt.ylabel("Max abs rolling market beta")
        plt.grid(alpha=0.3)
        out = NORMAL_BETA_HEATMAP.replace(".png", f"_scale_{signal_scale:g}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"[PLOT] Saved {out}")




def main() -> None:
    print("=" * 72)
    print("  Market-Neutral MVO Stress Test: Idio Signal Forecast-Error Noise")
    print("=" * 72)

    wide, symbols = load_data(DB_PATH, TOP_N)
    rets, mkt = compute_returns(wide)

    print(f"[INFO] Returns: {rets.shape[0]} days x {rets.shape[1]} stocks")
    print(f"[INFO] Dates: {rets.index[0].date()} -> {rets.index[-1].date()}")

    print("\n[OLS] Estimating rolling alpha/beta and idio returns...")
    alphas_df, betas_df, resid_df = rolling_ols(rets, mkt, WINDOW)
    print(f"[OLS] Non-NaN residuals: {resid_df.notna().sum().sum():,}")

    scenarios = build_scenarios()
    print(f"\n[SIM] Running {len(scenarios)} scenarios...")

    pnl_by_scenario: dict[int, pd.Series] = {}
    daily_diag_list: list[pd.DataFrame] = []
    summary_rows: list[dict] = []


    #TODO wrap in multiprocessing
    for scenario in tqdm(scenarios, desc="Scenarios", unit="scenario"):
        pnl_series, diagnostics, weights = run_scenario(
            scenario=scenario,
            rets=rets,
            mkt=mkt,
            betas_df=betas_df,
            resid_df=resid_df,
            window=WINDOW,
        )

        if len(pnl_series) == 0:
            continue

        pnl_by_scenario[scenario.scenario_id] = pnl_series
        daily_diag_list.append(diagnostics)
        summary_rows.append(compute_summary(scenario, pnl_series, diagnostics, mkt))

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[OUT] Saved {SUMMARY_CSV}")

    if daily_diag_list:
        daily_diag = pd.concat(daily_diag_list, ignore_index=True)
        daily_diag.to_csv(DAILY_DIAG_CSV, index=False)
        print(f"[OUT] Saved {DAILY_DIAG_CSV}")

    print("\n" + "=" * 72)
    print("  Top 10 scenarios by Sharpe")
    print("=" * 72)
    cols = [
        "scenario_id", "dist", "signal_scale", "noise_mean", "noise_std",
        "sharpe", "ann_ret", "ann_vol", "max_drawdown", "realized_beta",
        "max_abs_rolling_beta", "mean_turnover", "mean_signal_rank_ic",
    ]
    print(summary[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("  Worst 10 scenarios by Sharpe")
    print("=" * 72)
    print(summary[cols].tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("  Worst 10 scenarios by beta leakage")
    print("=" * 72)
    print(
        summary.sort_values("max_abs_rolling_beta", ascending=False)[cols]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n[PLOT] Generating charts...")
    plot_top_cumulative_pnl(pnl_by_scenario, summary)
    plot_normal_heatmaps(summary)

    print("\n[DONE] Stress test complete.")


if __name__ == "__main__":
    main()
