# Portfolio Lab: Modern Portfolio Theory in Python

An interactive portfolio-optimization dashboard that compares historical risk and return for a basket of publicly traded assets.

## Features
- Historical price retrieval using Yahoo Finance (`yfinance`)
- Annualized return, volatility, and Sharpe ratio
- Monte Carlo portfolio simulation
- Efficient-frontier-style risk/return scatter plot
- Maximum-Sharpe and minimum-volatility allocations
- Equal-weight baseline
- Normalized historical performance chart
- Downloadable allocation CSV

## Run locally (Windows PowerShell)

```powershell
cd portfolio_optimizer_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Suggested demo
Use `AAPL, MSFT, GOOGL, AMZN, NVDA` with a 5-year window. You can also try ETFs such as `SPY, QQQ, IWM, TLT, GLD`. Ticker availability depends on Yahoo Finance.

## How it works
Daily returns are computed from adjusted prices. Annualized mean return is approximated as mean daily return × 252. Annualized volatility is derived from the daily covariance matrix × 252. Random long-only weights are sampled from a Dirichlet distribution. SciPy's SLSQP optimizer estimates maximum-Sharpe and minimum-volatility portfolios, subject to weights between 0 and 100% and total weight equal to 100%.

## Important limitations
This is an educational analytics project, not financial advice. Results are sensitive to the selected dates and estimated returns. It does not include taxes, fees, liquidity constraints, or future forecasts. Past performance does not guarantee future results.

## Resume bullet (edit to match your actual work)
**Portfolio Optimization Dashboard | Python, Pandas, NumPy, SciPy, Plotly, Streamlit**
- Built an interactive Modern Portfolio Theory dashboard to simulate thousands of long-only portfolios, visualize risk-return trade-offs, and compare maximum-Sharpe, minimum-volatility, and equal-weight allocations.
- Implemented annualized return, volatility, and Sharpe ratio calculations with downloadable portfolio weights and historical performance visualization.

Only claim the project after running it, checking the results, and understanding the methodology.
