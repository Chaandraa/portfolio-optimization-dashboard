import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sqlite3  # Added to manage database interactions
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 Portfolio Lab")
st.caption("Modern Portfolio Theory • Risk-return analysis • Efficient frontier")
st.info("Educational project only. Outputs are historical estimates, not investment advice or guaranteed future performance.")

# --- NEW STEP: CORE DATABASE PIPELINE FUNCTION ---
def load_data_from_db():
    """Extracts the entire historical stock price table using a SQL query."""
    try:
        # Connect to your newly built local relational database
        conn = sqlite3.connect('portfolio.db')
        
        # Standard SQL Query to fetch all records
        query = "SELECT * FROM historical_prices"
        df = pd.read_sql_query(query, conn)
        
        # Format the Date index columns cleanly for downstream analytics
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Read Failure: {e}")
        st.info("Make sure you ran 'python init_db.py' first to initialize your local portfolio.db file.")
        st.stop()

# Load our historical data using local SQL rather than querying the internet directly
all_historical_data = load_data_from_db()

# Dynamically extract available stock tickers present inside our Database column headers
available_tickers = [col for col in all_historical_data.columns]

with st.sidebar:
    st.header("Analysis settings")
    # Upgraded UI component: Instead of a text input box, select tickers directly from what's stored in your SQL database!
    selected_tickers = st.multiselect("Select Tickers from Database", options=available_tickers, default=available_tickers)
    risk_free_rate = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.25) / 100
    simulations = st.slider("Random portfolios", min_value=1000, max_value=20000, value=5000, step=1000)
    run = st.button("Run portfolio analysis", type="primary", use_container_width=True)

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = True
if run:
    st.session_state.run_analysis = True

def portfolio_stats(weights, mean_returns, cov):
    ret = float(np.dot(mean_returns, weights) * 252)
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov * 252, weights))))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
    return ret, vol, sharpe

def optimize(mean_returns, cov, objective):
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.repeat(1 / n, n)
    if objective == "sharpe":
        fun = lambda w: -portfolio_stats(w, mean_returns, cov)[2]
    else:
        fun = lambda w: portfolio_stats(w, mean_returns, cov)[1]
    result = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else x0

if st.session_state.run_analysis:
    if len(selected_tickers) < 2:
        st.error("Select at least two ticker symbols.")
        st.stop()
    
    with st.spinner("Querying SQLite Database and calculating portfolios..."):
        try:
            # Filter our local master database DataFrame based on the user's sidebar selection
            prices = all_historical_data[selected_tickers].dropna()
            
            if prices.shape[1] < 2 or len(prices) < 60:
                raise ValueError("Not enough overlapping price history available for these specific assets.")
                
            returns = prices.pct_change().dropna()
            mean_returns = returns.mean().values
            cov = returns.cov().values
            
        except Exception as e:
            st.error(f"Could not prepare price data: {e}")
            st.stop()

    n = len(prices.columns)
    rng = np.random.default_rng(42)
    weights_matrix = rng.dirichlet(np.ones(n), size=simulations)
    annual_returns = weights_matrix @ mean_returns * 252
    annual_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights_matrix, cov * 252, weights_matrix))
    sharpes = np.divide(annual_returns - risk_free_rate, annual_vols, out=np.zeros_like(annual_returns), where=annual_vols > 0)

    max_sharpe_w = optimize(mean_returns, cov, "sharpe")
    min_vol_w = optimize(mean_returns, cov, "vol")
    max_ret, max_vol, max_sharpe = portfolio_stats(max_sharpe_w, mean_returns, cov)
    min_ret, min_vol, min_sharpe = portfolio_stats(min_vol_w, mean_returns, cov)
    eq_w = np.repeat(1/n, n)
    eq_ret, eq_vol, eq_sharpe = portfolio_stats(eq_w, mean_returns, cov)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets analyzed", len(prices.columns))
    c2.metric("Max-Sharpe expected return", f"{max_ret:.1%}")
    c3.metric("Max-Sharpe volatility", f"{max_vol:.1%}")
    c4.metric("Max-Sharpe ratio", f"{max_sharpe:.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Efficient frontier", "Recommended allocations", "Performance", "Data & methodology"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=annual_vols, y=annual_returns, mode="markers",
                                 marker=dict(size=5, color=sharpes, colorscale="Viridis", showscale=True,
                                             colorbar=dict(title="Sharpe")),
                                 name="Random portfolios", hovertemplate="Volatility: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[max_vol], y=[max_ret], mode="markers+text",
                                 marker=dict(size=15, symbol="star", color="red"),
                                 text=["Max Sharpe"], textposition="top center", name="Max Sharpe"))
        fig.add_trace(go.Scatter(x=[min_vol], y=[min_ret], mode="markers+text",
                                 marker=dict(size=12, symbol="diamond", color="blue"),
                                 text=["Min volatility"], textposition="bottom right", name="Min volatility"))
        fig.add_trace(go.Scatter(x=[eq_vol], y=[eq_ret], mode="markers+text",
                                 marker=dict(size=11, symbol="circle", color="black"),
                                 text=["Equal weight"], textposition="bottom left", name="Equal weight"))
        fig.update_layout(title="Simulated portfolios: annualized risk vs. return",
                          xaxis_title="Annualized volatility (standard deviation)",
                          yaxis_title="Annualized historical return", yaxis_tickformat=".0%",
                          xaxis_tickformat=".0%", template="plotly_white", height=560)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot is a long-only portfolio whose weights sum to 100%. Historical mean returns are annualized using 252 trading days.")
    with tab2:
        alloc = pd.DataFrame({
            "Ticker": list(prices.columns),
            "Max-Sharpe Weight": max_sharpe_w,
            "Min-Volatility Weight": min_vol_w,
            "Equal-Weight": eq_w
        })
        display = alloc.copy()
        for col in display.columns[1:]:
            display[col] = display[col].map(lambda x: f"{x:.1%}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("Download allocations CSV", alloc.to_csv(index=False).encode("utf-8"),
                           file_name="portfolio_allocations.csv", mime="text/csv")
        st.warning("Optimized weights can be unstable when expected returns are estimated from limited historical data. Treat them as a learning exercise.")
    with tab3:
        norm_prices = prices / prices.iloc[0] * 100
        fig2 = px.line(norm_prices, x=norm_prices.index, y=norm_prices.columns,
                       title="Growth of 100 units invested (normalized prices)")
        fig2.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Value (start = 100)")
        st.plotly_chart(fig2, use_container_width=True)
        metric_data = pd.DataFrame([
            ["Max Sharpe", max_ret, max_vol, max_sharpe],
            ["Minimum volatility", min_ret, min_vol, min_sharpe],
            ["Equal weight", eq_ret, eq_vol, eq_sharpe],
        ], columns=["Portfolio", "Annualized return", "Annualized volatility", "Sharpe ratio"])
        st.dataframe(metric_data.style.format({"Annualized return":"{:.2%}", "Annualized volatility":"{:.2%}", "Sharpe ratio":"{:.2f}"}), use_container_width=True, hide_index=True)
    with tab4:
        st.markdown("""
**Methodology & System Architecture**
1. An independent ETL script (`init_db.py`) extracts 10 years of adjusted market closings and maps them into a local relational `SQLite` database database file (`portfolio.db`).
2. The Streamlit application consumes data explicitly out of local tables via **optimized SQL engine commands** using `pandas.read_sql_query`, bypassing rate limits.
3. Daily mathematical statistics and covariance percentage configurations are handled directly through vectorizations.
4. Constrained optimizations calculate maximum-Sharpe bounds and minimum-volatility allocations via `SciPy` SLSQP methodologies.
""")
        st.subheader("Database live data preview")
        st.dataframe(prices.tail(10), use_container_width=True)
else:
    st.write("Set your tickers in the sidebar and click **Run portfolio analysis**.")
