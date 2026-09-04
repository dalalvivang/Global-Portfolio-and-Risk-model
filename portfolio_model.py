import sys
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t
from sklearn.covariance import LedoitWolf
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 2. CUSTOM EXCEPTIONS AND CONSTANTS
class DataError(Exception):
    """Raised when data acquisition fails."""
    pass

class FXError(Exception):
    """Raised when currency data acquisition fails."""
    pass

CURRENCY_MAP = {
    ".NS": "INR", ".SW": "CHF", ".L": "GBP", ".PA": "EUR",
    ".DE": "EUR", ".AX": "AUD", ".MC": "EUR", ".SI": "SGD", ".TO": "CAD"
}
MANUAL_OVERRIDES = {"TRI": "CAD", "DBF.AX": "AUD"}

TICKERS = [
    "RTX", "BEL.NS", "LMT", "BA.L", "SAF.PA", "SUNPHARMA.NS", "JNJ", "PFE", "ZTS",
    "TATACONSUM.NS", "ITC.NS", "NESN.SW", "DGE.L", "AAC.AX", "SIE.DE", "TATAPOWER.NS",
    "TTE.PA", "IBE.MC", "NEE", "TMPV.NS", "VOW3.DE", "DMART.NS", "COST", "INDHOTEL.NS",
    "MC.PA", "ADANIPORTS.NS", "ACS.MC", "LT.NS", "GOOGL", "MSFT", "ORCL", "HDFCBANK.NS",
    "BLK", "D05.SI", "UBS", "BHP.AX", "RIO.AX", "TRI", "DBF.AX"
]
BENCHMARK_TICKERS = {"MSCI World": "URTH", "S&P 500": "SPY"}
START_DATE = "2015-01-01"
RF_ANNUAL = 0.045
BASE_CURRENCY = 'EUR'
MIN_WEIGHT = 0.00
MAX_WEIGHT = 0.20


def get_currency(ticker: str) -> str:
    """Resolve the settlement currency for a given ticker."""
    if ticker in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[ticker]
    for suffix, curr in CURRENCY_MAP.items():
        if ticker.endswith(suffix):
            return curr
    return "USD"


def var_95(rets: pd.Series) -> float:
    """Value at Risk: magnitude of the 5th percentile monthly loss."""
    return rets.quantile(0.05)


def cvar_95(rets: pd.Series) -> float:
    """Conditional VaR (Expected Shortfall)."""
    cutoff = var_95(rets)
    return rets[rets <= cutoff].mean()


def series_stats(rets: pd.Series) -> tuple:
    """Compute risk/reward metrics from a pre-computed return series."""
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else 0
    mar_m = RF_ANNUAL / 12
    shortfall = rets[rets < mar_m] - mar_m
    downside_vol = np.sqrt(np.mean(shortfall**2)) * np.sqrt(12) if not shortfall.empty else 1e-6
    sortino = (ann_ret - RF_ANNUAL) / downside_vol
    cum = (1 + rets).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    return ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, var_95(rets), cvar_95(rets)


def p_stats(w: np.ndarray, returns: pd.DataFrame, cov_matrix: pd.DataFrame = None) -> tuple:
    """Compute portfolio-level risk/reward metrics for a weight vector."""
    p_rets = returns.dot(w)
    return series_stats(p_rets)


# 3. DATA ACQUISITION AND FX CONVERSION
try:
    logging.info("Acquiring market data...")
    raw = yf.download(TICKERS + list(BENCHMARK_TICKERS.values()), start=START_DATE, auto_adjust=True)['Close']
    data_m = raw[TICKERS].resample('ME').last()
    bench_m = raw[list(BENCHMARK_TICKERS.values())].resample('ME').last()
except Exception as e:
    logging.error(f"Data acquisition failed: {e}")
    raise DataError(str(e)) from e

adj_data = pd.DataFrame(index=data_m.index)

for t_sym in data_m.columns:
    curr = get_currency(t_sym)
    local_price = data_m[t_sym]
    if curr == BASE_CURRENCY:
        adj_data[t_sym] = local_price
    else:
        try:
            fx = yf.download(f"{curr}{BASE_CURRENCY}=X", start=START_DATE, auto_adjust=True)['Close']
            if isinstance(fx, pd.DataFrame):
                fx = fx.iloc[:, 0]
            fx_aligned = fx.resample('ME').last().reindex(data_m.index).ffill()
        except Exception:
            logging.warning(f"FX data unavailable for {curr}. Using 1.0.")
            fx_aligned = pd.Series(1.0, index=data_m.index)
        adj_data[t_sym] = local_price * fx_aligned

data_final = adj_data.ffill().dropna(axis=1, thresh=24)
rets_m = data_final.pct_change().dropna()
lw = LedoitWolf().fit(rets_m)
cov_shrunk = pd.DataFrame(lw.covariance_ * 12, index=rets_m.columns, columns=rets_m.columns)


# 4. ANALYTICS ENGINE
def monte_carlo_simulation(weights: np.ndarray, rets_m: pd.DataFrame, n_sims: int = 1000, n_months: int = 60) -> np.ndarray:
    """Generate 1000 future portfolio-level return paths using Student-t(df=5)."""
    port_rets = rets_m.dot(weights)
    mu = port_rets.mean()
    sigma = port_rets.std()
    raw = t.rvs(df=5, size=(n_months, n_sims))
    scaled = mu + sigma * raw / np.sqrt(5 / (5 - 2))
    sims = np.exp(scaled).cumprod(axis=0)
    return sims


# 5. WALK-FORWARD BACKTEST
def walk_forward_backtest(rets: pd.DataFrame, est_window: int = 36, test_window: int = 12) -> pd.Series:
    """Run a rolling walk-forward Sharpe-maximising backtest and return OOS returns."""
    oos_results = []
    for i in range(0, len(rets) - est_window - test_window + 1, test_window):
        train = rets.iloc[i: i + est_window]
        test = rets.iloc[i + est_window: i + est_window + test_window]
        assert train.columns.equals(test.columns), f"Column mismatch in window starting {i}"
        try:
            lw_t = LedoitWolf().fit(train)
            c_sh = pd.DataFrame(lw_t.covariance_ * 12, index=train.columns, columns=train.columns)
            res = minimize(lambda w: -p_stats(w, train, c_sh)[2], x0=np.array([1. / len(train.columns)] * len(train.columns)),
                            bounds=tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(len(train.columns))),
                            constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))
            oos_results.append(test.dot(res.x))
        except Exception:
            oos_results.append(test.mean(axis=1))
    return pd.concat(oos_results)


# 6. PORTFOLIO OPTIMISATION
EST_WINDOW = 36  # must match walk_forward_backtest's est_window

def optimize_weights(rets: pd.DataFrame, cov: pd.DataFrame) -> np.ndarray:
    """Max-Sharpe optimization, factored out so IS and full-sample fits use identical logic."""
    n = len(rets.columns)
    result = minimize(
        lambda w: -p_stats(w, rets, cov)[2],
        x0=np.array([1. / n] * n),
        bounds=tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)),
        constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    )
    return result.x

# "Current" recommended portfolio: fit on the FULL sample.
# Used for the weights table and the forward-looking Monte Carlo —
# this is a legitimate use of all available history for a live
# recommendation, and is NOT the fair walk-forward comparison.
port_weights = optimize_weights(rets_m, cov_shrunk)

# Fair in-sample comparison portfolio: fit ONLY on the first
# walk-forward training window. This has no look-ahead into any
# period used in oos_rets, unlike a full-sample fit — making it
# the correct "ceiling" to compare against walk-forward OOS.
is_train = rets_m.iloc[:EST_WINDOW]
is_cov = pd.DataFrame(
    LedoitWolf().fit(is_train).covariance_ * 12,
    index=is_train.columns, columns=is_train.columns
)
is_weights = optimize_weights(is_train, is_cov)

oos_rets = walk_forward_backtest(rets_m)
port_rets = rets_m.dot(port_weights)       # full-sample portfolio, for reference only
is_port_rets = rets_m.dot(is_weights)      # fair IS comparison line
mc_paths = monte_carlo_simulation(port_weights, rets_m)


# 7. DASHBOARD (6 panels)
fig = plt.figure(figsize=(22, 18))

ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax1.plot((1 + is_port_rets).cumprod(), label='In-Sample Max Sharpe (fair, first-window fit)', color='#2c3e50', lw=2)
ax1.plot((1 + oos_rets).cumprod(), label='Walk-Forward OOS', color='#27ae60', lw=3)
ax1.plot((1 + port_rets).cumprod(), label='Full-Sample Fit (look-ahead, reference only)', color='#95a5a6', lw=1, ls=':')  # remove this line to drop the reference series
ax1.axvline(oos_rets.index[0], color='gray', ls=':', lw=1, label='OOS start')
for n, s in BENCHMARK_TICKERS.items():
    ax1.plot((1 + bench_m[s].pct_change().dropna()).cumprod(), label=f'Bench: {n}', alpha=0.4, ls='--')
ax1.set_title("Portfolio Performance Validation")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot2grid((3, 2), (1, 0))
sns.heatmap(rets_m.corr(), cmap='RdYlGn', ax=ax2)
ax2.set_title("Asset Correlation Matrix")

ax3 = plt.subplot2grid((3, 2), (1, 1))
weights_s = pd.Series(port_weights, index=rets_m.columns).sort_values().tail(10)
weights_s.plot(kind='barh', ax=ax3, title="Top 10 Portfolio Weights (Full-Sample Fit)")

ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
is_port_rets.rolling(12).std().multiply(np.sqrt(12)).plot(ax=ax4, label='IS Rolling Vol (12M, fair)', color='blue')
oos_rets.rolling(12).std().multiply(np.sqrt(12)).plot(ax=ax4, label='OOS Rolling Vol (12M)', color='green')
ax4.set_title("Rolling Annualized Volatility (12-Month)")
ax4.legend()
ax4.grid(True)

ax5 = plt.subplot2grid((3, 2), (2, 1))
for path in mc_paths.T[:100]:
    ax5.plot(path, alpha=0.02, color='blue')
ax5.set_title("Monte Carlo: 60-month paths (Student-t, 1000 simulations)")
ax5.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()


# 8. SUMMARY TABLES AND STRESS TESTS
print("\n--- OPTIMIZED WEIGHTS (Full-Sample Fit) ---")
print(pd.Series(port_weights, index=rets_m.columns).sort_values(ascending=False).round(4).head(15))

print("\n--- RISK/REWARD METRICS (Including VaR-95) ---")
metrics = ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Sortino', 'Max DD', 'Calmar', 'VaR-95', 'CVaR-95']
report = {
    "Portfolio (IS, fair)": series_stats(is_port_rets),
    "Portfolio (Full-Sample, reference)": series_stats(port_rets),
    "Portfolio (OOS)": series_stats(oos_rets),
}
for n, s in BENCHMARK_TICKERS.items():
    report[n] = series_stats(bench_m[s].pct_change().dropna())
print(pd.DataFrame(report, index=metrics).T.round(4))

print("\n--- STRESS PERIOD PERFORMANCE ---")
stress_periods = {
    "COVID crash": ("2020-02-01", "2020-03-31"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
    "China devaluation": ("2015-08-01", "2015-09-30"),
}
urth_aligned = bench_m[BENCHMARK_TICKERS["MSCI World"]].pct_change().reindex(rets_m.index)
spy_aligned = bench_m[BENCHMARK_TICKERS["S&P 500"]].pct_change().reindex(rets_m.index)

stress_results = []
for name, (start, end) in stress_periods.items():
    mask = (rets_m.index >= start) & (rets_m.index <= end)
    is_ret_fair = rets_m[mask].dot(is_weights).sum()
    full_ret = rets_m[mask].dot(port_weights).sum()
    oos_mask = (oos_rets.index >= start) & (oos_rets.index <= end)
    oos_ret = oos_rets[oos_mask].sum()
    urth_ret = urth_aligned[mask].sum()
    spy_ret = spy_aligned[mask].sum()
    stress_results.append({
        "Period": name,
        "IS Return (fair)": is_ret_fair,
        "Full-Sample Return (reference)": full_ret,
        "OOS Return": oos_ret,
        "URTH": urth_ret,
        "SPY": spy_ret,
    })
print(pd.DataFrame(stress_results).round(4))
