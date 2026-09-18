import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ETF Rotation Backtester", layout="wide")

st.title("📊 Multi-Asset ETF Rotation Strategy Backtester")

# Sidebar Configuration
st.sidebar.header("1. Asset Universe & Proxies")
use_proxies = st.sidebar.checkbox("Use Pre-2000 Mutual Fund Proxies", value=True, 
                                  help="Swaps SPY->VFINX, TLT->VUSTX, EEM->VEIEX for long history backtesting.")

default_universe = "VFINX, VUSTX, VEIEX, EWJ" if use_proxies else "SPY, TLT, EEM, GLD, EWJ, EWZ, FXI"
ticker_input = st.sidebar.text_input("Asset Universe (Comma-separated)", value=default_universe)

benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="VFINX" if use_proxies else "SPY")
defensive_ticker = st.sidebar.text_input("Defensive / Risk-Off Asset", value="VUSTX" if use_proxies else "TLT")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("1994-01-01") if use_proxies else pd.to_datetime("2004-11-18"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.header("2. Strategy Parameters")
lookback_months = st.sidebar.slider("Momentum Lookback (Months)", 1, 12, 6)
top_n = st.sidebar.slider("Top N Assets to Hold", 1, 4, 1)
rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["Monthly", "Quarterly"])

use_regime = st.sidebar.checkbox("Enable Risk-On / Risk-Off Regime Filter", value=True)
regime_sma_days = st.sidebar.slider("Benchmark Trend Filter (SMA Days)", 50, 250, 200)

st.sidebar.header("3. Execution Costs")
tx_cost_bps = st.sidebar.number_input("Transaction Fee / Slippage per Trade (bps)", 0.0, 50.0, 5.0) / 10000

# Robust Market Data Downloader
@st.cache_data(ttl=3600)
def load_market_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.levels[0]:
            data = df['Close']
        elif 'Adj Close' in df.columns.levels[0]:
            data = df['Adj Close']
        else:
            data = df.xs(df.columns.levels[0][0], axis=1, level=0)
    else:
        if 'Close' in df.columns:
            data = df['Close']
        elif 'Adj Close' in df.columns:
            data = df['Adj Close']
        else:
            data = df

    if isinstance(data, pd.Series):
        data = data.to_frame()
        
    return data.ffill().dropna(how='all')

tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
all_tickers = list(set(tickers_list + [benchmark_ticker, defensive_ticker]))

with st.spinner("Downloading historical price data..."):
    df_prices = load_market_data(all_tickers, start_date, end_date)

if df_prices.empty or benchmark_ticker not in df_prices.columns:
    st.error("Error loading data. Verify ticker symbols and date ranges.")
    st.stop()

# Data Resampling & Momentum Signals
freq_code = 'ME' if rebalance_freq == "Monthly" else 'QE'
df_reb_prices = df_prices.resample(freq_code).last()

# Calculate Lookback Returns (Momentum)
df_momentum = df_reb_prices[tickers_list].pct_change(periods=lookback_months)

# Calculate Benchmark Moving Average
df_prices['Bench_SMA'] = df_prices[benchmark_ticker].rolling(window=regime_sma_days).mean()
reb_regime = df_prices['Bench_SMA'].resample(freq_code).last()

# Backtest Engine
returns_df = df_prices.pct_change().fillna(0)
portfolio_weights = pd.DataFrame(0.0, index=df_reb_prices.index, columns=tickers_list + [defensive_ticker])

for idx in range(lookback_months, len(df_reb_prices)):
    date = df_reb_prices.index[idx]
    
    # Check regime condition
    bench_price = df_reb_prices.loc[date, benchmark_ticker] if benchmark_ticker in df_reb_prices.columns else 0
    bench_sma = reb_regime.loc[date] if not pd.isna(reb_regime.loc[date]) else 0
    
    regime_risk_on = (bench_price >= bench_sma) if use_regime else True
    
    if regime_risk_on:
        # Rank momentum among available universe assets
        mom_row = df_momentum.loc[date].dropna()
        valid_assets = mom_row[mom_row > 0]  # Only hold assets with positive momentum
        
        if len(valid_assets) > 0:
            selected = valid_assets.nlargest(top_n).index.tolist()
            weight_per_asset = 1.0 / len(selected)
            for asset in selected:
                portfolio_weights.loc[date, asset] = weight_per_asset
        else:
            # Fallback to defensive asset if no asset has positive momentum
            portfolio_weights.loc[date, defensive_ticker] = 1.0
    else:
        # Risk-off: allocate 100% to defensive asset
        portfolio_weights.loc[date, defensive_ticker] = 1.0

# Daily Portfolio Tracking
daily_weights = portfolio_weights.reindex(df_prices.index).ffill().fillna(0)

# Calculate Turnover & Costs
weight_changes = daily_weights.diff().abs().sum(axis=1)
daily_tx_costs = weight_changes * tx_cost_bps

daily_asset_returns = returns_df[daily_weights.columns]
strategy_daily_returns = (daily_weights * daily_asset_returns).sum(axis=1) - daily_tx_costs
benchmark_daily_returns = returns_df[benchmark_ticker]

# Equity Curves
strat_equity = (1 + strategy_daily_returns).cumprod()
bench_equity = (1 + benchmark_daily_returns).cumprod()

# Performance Analytics Function
def calc_metrics(daily_returns, equity_curve):
    total_years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 0.1)
    cagr = (equity_curve.iloc[-1] ** (1 / total_years)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() * 252 - 0.02) / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    
    # Sortino
    downside_vol = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino = (daily_returns.mean() * 252 - 0.02) / downside_vol if downside_vol > 0 else 0
    
    return {
        "CAGR": cagr,
        "Volatility": ann_vol,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino
    }

strat_metrics = calc_metrics(strategy_daily_returns, strat_equity)
bench_metrics = calc_metrics(benchmark_daily_returns, bench_equity)

# Display Key Metrics Table
st.header("Strategy vs Benchmark Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Strategy CAGR", f"{strat_metrics['CAGR']:.2%}", f"{(strat_metrics['CAGR'] - bench_metrics['CAGR']):.2%} vs Bench")
col2.metric("Max Drawdown", f"{strat_metrics['Max Drawdown']:.2%}", f"{(strat_metrics['Max Drawdown'] - bench_metrics['Max Drawdown']):.2%} vs Bench")
col3.metric("Sharpe Ratio", f"{strat_metrics['Sharpe Ratio']:.2f}", f"{(strat_metrics['Sharpe Ratio'] - bench_metrics['Sharpe Ratio']):.2f}")
col4.metric("Sortino Ratio", f"{strat_metrics['Sortino Ratio']:.2f}")
col5.metric("Ann. Volatility", f"{strat_metrics['Volatility']:.2%}")

# Interactive Charts
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

fig.add_trace(go.Scatter(x=strat_equity.index, y=strat_equity.values, name="Rotation Strategy", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=bench_equity.index, y=bench_equity.values, name=f"Buy & Hold ({benchmark_ticker})", line=dict(color="#7f7f7f", width=1.5, dash="dash")), row=1, col=1)

# Drawdowns
strat_dd = (strat_equity - strat_equity.cummax()) / strat_equity.cummax()
bench_dd = (bench_equity - bench_equity.cummax()) / bench_equity.cummax()

fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values, name="Strategy Drawdown", fill='tozeroy', line=dict(color="#d62728", width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, name="Benchmark Drawdown", line=dict(color="gray", width=1, dash="dot")), row=2, col=1)

fig.update_layout(title="Cumulative Return & Drawdown Analysis (Log Scale)", yaxis_type="log", height=650, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# Historical Holding Distribution
st.header("Historical Asset Allocation")
st.area_chart(daily_weights)
