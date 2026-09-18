import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Advanced ETF Timing Engine", layout="wide")

st.title("📊 Multi-Strategy ETF Timing & Backtester Engine")

# --- 1. ASSET UNIVERSE & DATA SETUP ---
st.sidebar.header("1. Universe & Data")
use_proxies = st.sidebar.checkbox("Use Pre-2000 Mutual Fund Proxies", value=True, 
                                  help="Swaps SPY->VFINX, TLT->VUSTX, EEM->VEIEX for 1990s backtesting.")

default_universe = "VFINX, VUSTX, VEIEX, EWJ" if use_proxies else "SPY, TLT, EEM, GLD, EWJ, EWZ, FXI"
ticker_input = st.sidebar.text_input("Asset Universe (Comma-separated)", value=default_universe)

benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="VFINX" if use_proxies else "SPY").strip().upper()

defensive_option = st.sidebar.selectbox("Defensive / Risk-Off Asset Choice", ["Cash (0% Return)", "Custom Ticker Symbol"])
if defensive_option == "Custom Ticker Symbol":
    defensive_ticker = st.sidebar.text_input("Defensive Ticker", value="VUSTX" if use_proxies else "TLT").strip().upper()
else:
    defensive_ticker = "CASH"

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("1994-01-01") if use_proxies else pd.to_datetime("2004-11-18"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- 2. TIMING STRATEGY MODULES ---
st.sidebar.header("2. Momentum & Selection Engine")
mom_type = st.sidebar.selectbox("Momentum Calculation", ["Single Window", "Dual Window Average (3M + 6M)"])
lookback_months = st.sidebar.slider("Single Lookback (Months)", 1, 12, 6) if mom_type == "Single Window" else 6
top_n = st.sidebar.slider("Top N Assets to Hold", 1, 4, 1)
enable_abs_mom = st.sidebar.checkbox("Enable Absolute Momentum Filter (GEM)", value=True, help="Requires asset momentum > 0% to be eligible.")

st.sidebar.header("3. Macro & Trend Filter")
trend_type = st.sidebar.selectbox("Regime Trend Type", ["Standard Moving Average (SMA)", "Kaufman Adaptive Moving Average (KAMA)", "None / Always Risk-On"])
trend_period = st.sidebar.slider("Trend Filter Period (Days)", 20, 250, 200) if trend_type != "None / Always Risk-On" else 200

st.sidebar.header("4. Volatility Targeting & Risk Control")
enable_vol_target = st.sidebar.checkbox("Enable Volatility Targeting", value=False)
target_vol_pct = st.sidebar.slider("Target Annualized Volatility (%)", 5, 25, 12) / 100.0 if enable_vol_target else 0.12
vol_lookback_days = st.sidebar.slider("Volatility Estimate Window (Days)", 10, 90, 20) if enable_vol_target else 20

st.sidebar.header("5. Execution & Costs")
rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["Monthly", "Quarterly"])
tx_cost_bps = st.sidebar.number_input("Transaction Fee / Slippage per Trade (bps)", 0.0, 50.0, 5.0) / 10000

# Helper Function: KAMA Calculation
def calc_kama(series, n=10, pow1=2, pow2=30):
    change = (series - series.shift(n)).abs()
    volatility = (series - series.shift(1)).abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(pow1+1) - 2/(pow2+1)) + 2/(pow2+1)) ** 2
    kama = pd.Series(index=series.index, dtype='float64')
    
    # Initialize with SMA where available
    kama_init = series.rolling(n).mean()
    if kama_init.first_valid_index() is None:
        return series
    first_idx = series.index.get_loc(kama_init.first_valid_index())
    kama.iloc[first_idx] = kama_init.iloc[first_idx]
    
    for i in range(first_idx + 1, len(series)):
        val = sc.iloc[i]
        if np.isnan(val):
            val = 0
        kama.iloc[i] = kama.iloc[i-1] + val * (series.iloc[i] - kama.iloc[i-1])
    return kama

# Market Data Loader
@st.cache_data(ttl=3600)
def load_market_data(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        data = df['Close'] if 'Close' in df.columns.levels[0] else df['Adj Close']
    else:
        data = df['Close'] if 'Close' in df.columns else df.get('Adj Close', df)

    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.ffill().dropna(how='all')

tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
download_tickers = list(set([t for t in tickers_list + [benchmark_ticker, defensive_ticker] if t and t != "CASH"]))

with st.spinner("Fetching market history..."):
    df_prices = load_market_data(download_tickers, start_date, end_date)

if defensive_ticker == "CASH":
    df_prices["CASH"] = 1.0

if df_prices.empty or benchmark_ticker not in df_prices.columns:
    st.error("Data download failed. Check ticker symbols and date range.")
    st.stop()

# --- BACKTEST COMPUTATION ENGINE ---
freq_code = 'ME' if rebalance_freq == "Monthly" else 'QE'
df_reb_prices = df_prices.resample(freq_code).last()
universe_tickers = [t for t in tickers_list if t in df_reb_prices.columns and t != "CASH"]

# 1. Momentum Signal Matrix
if mom_type == "Single Window":
    df_momentum = df_reb_prices[universe_tickers].pct_change(periods=lookback_months)
else:
    mom_3m = df_reb_prices[universe_tickers].pct_change(periods=3)
    mom_6m = df_reb_prices[universe_tickers].pct_change(periods=6)
    df_momentum = (mom_3m + mom_6m) / 2.0

# 2. Benchmark Trend Filter Calculation
if trend_type == "Standard Moving Average (SMA)":
    df_prices['Bench_Trend'] = df_prices[benchmark_ticker].rolling(window=trend_period).mean()
elif trend_type == "Kaufman Adaptive Moving Average (KAMA)":
    df_prices['Bench_Trend'] = calc_kama(df_prices[benchmark_ticker], n=trend_period)
else:
    df_prices['Bench_Trend'] = 0.0  # Always active

reb_trend = df_prices['Bench_Trend'].resample(freq_code).last()

# 3. Rebalance Weight Allocation Engine
all_assets = list(dict.fromkeys(universe_tickers + [defensive_ticker]))
portfolio_weights = pd.DataFrame(0.0, index=df_reb_prices.index, columns=all_assets)

start_idx = 6 if mom_type == "Dual Window Average (3M + 6M)" else lookback_months

for idx in range(start_idx, len(df_reb_prices)):
    date = df_reb_prices.index[idx]
    
    # Check Macro Regime
    bench_p = df_reb_prices.loc[date, benchmark_ticker]
    trend_p = reb_trend.loc[date]
    risk_on_signal = (bench_p >= trend_p) if trend_type != "None / Always Risk-On" else True
    
    if risk_on_signal:
        mom_row = df_momentum.loc[date].dropna()
        valid_assets = mom_row[mom_row > 0] if enable_abs_mom else mom_row
        
        if len(valid_assets) > 0:
            selected = valid_assets.nlargest(top_n).index.tolist()
            w_per_asset = 1.0 / len(selected)
            for asset in selected:
                portfolio_weights.loc[date, asset] = w_per_asset
        else:
            portfolio_weights.loc[date, defensive_ticker] = 1.0
    else:
        portfolio_weights.loc[date, defensive_ticker] = 1.0

# 4. Expand Weights to Daily & Apply Volatility Targeting
daily_weights = portfolio_weights.reindex(df_prices.index).ffill().fillna(0)
returns_df = df_prices.pct_change().fillna(0)

if enable_vol_target:
    # Compute rolling portfolio realized volatility
    raw_daily_returns = (daily_weights * returns_df[daily_weights.columns]).sum(axis=1)
    rolling_vol = raw_daily_returns.rolling(window=vol_lookback_days).std() * np.sqrt(252)
    vol_scalar = (target_vol_pct / rolling_vol).clip(upper=1.0).fillna(1.0)
    
    # Scale risky assets, allocate remainder to defensive asset
    risky_cols = [c for c in daily_weights.columns if c != defensive_ticker]
    daily_weights[risky_cols] = daily_weights[risky_cols].mul(vol_scalar, axis=0)
    daily_weights[defensive_ticker] = 1.0 - daily_weights[risky_cols].sum(axis=1)

# 5. Turnover & Strategy Return Calculation
weight_changes = daily_weights.diff().abs().sum(axis=1)
daily_tx_costs = weight_changes * tx_cost_bps
strategy_daily_returns = (daily_weights * returns_df[daily_weights.columns]).sum(axis=1) - daily_tx_costs
benchmark_daily_returns = returns_df[benchmark_ticker]

strat_equity = (1 + strategy_daily_returns).cumprod()
bench_equity = (1 + benchmark_daily_returns).cumprod()

# --- PERFORMANCE ANALYTICS ---
def calc_metrics(daily_returns, equity_curve):
    total_years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 0.1)
    cagr = (equity_curve.iloc[-1] ** (1 / total_years)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() * 252 - 0.02) / ann_vol if ann_vol > 0 else 0
    peak = equity_curve.cummax()
    max_dd = ((equity_curve - peak) / peak).min()
    downside_vol = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino = (daily_returns.mean() * 252 - 0.02) / downside_vol if downside_vol > 0 else 0
    return {"CAGR": cagr, "Volatility": ann_vol, "Max Drawdown": max_dd, "Sharpe Ratio": sharpe, "Sortino Ratio": sortino}

strat_m = calc_metrics(strategy_daily_returns, strat_equity)
bench_m = calc_metrics(benchmark_daily_returns, bench_equity)

st.header("Strategy vs Benchmark Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Strategy CAGR", f"{strat_m['CAGR']:.2%}", f"{(strat_m['CAGR'] - bench_m['CAGR']):.2%} vs Bench")
c2.metric("Max Drawdown", f"{strat_m['Max Drawdown']:.2%}", f"{(strat_m['Max Drawdown'] - bench_m['Max Drawdown']):.2%} vs Bench")
c3.metric("Sharpe Ratio", f"{strat_m['Sharpe Ratio']:.2f}", f"{(strat_m['Sharpe Ratio'] - bench_m['Sharpe Ratio']):.2f}")
c4.metric("Sortino Ratio", f"{strat_m['Sortino Ratio']:.2f}")
c5.metric("Ann. Volatility", f"{strat_m['Volatility']:.2%}")

# Interactive Equity & Drawdown Charts
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Scatter(x=strat_equity.index, y=strat_equity.values, name="Combined Timing Strategy", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=bench_equity.index, y=bench_equity.values, name=f"Buy & Hold ({benchmark_ticker})", line=dict(color="#7f7f7f", width=1.5, dash="dash")), row=1, col=1)

strat_dd = (strat_equity - strat_equity.cummax()) / strat_equity.cummax()
bench_dd = (bench_equity - bench_equity.cummax()) / bench_equity.cummax()

fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values, name="Strategy Drawdown", fill='tozeroy', line=dict(color="#d62728", width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, name="Benchmark Drawdown", line=dict(color="gray", width=1, dash="dot")), row=2, col=1)

fig.update_layout(title="Cumulative Performance & Drawdown (Log Scale)", yaxis_type="log", height=650, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

st.header("Historical Asset Allocation")
st.area_chart(daily_weights)
