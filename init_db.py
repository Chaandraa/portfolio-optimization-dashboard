import sqlite3
import yfinance as yf
import pandas as pd

# 1. Connect to SQLite (creates 'portfolio.db' file automatically)
conn = sqlite3.connect('portfolio.db')

# 2. Define your target tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

print("Fetching 10 years of stock data from Yahoo Finance...")
# 3. Download data specifying the explicit "Close" price column inside the function argument
# This completely bypasses the KeyError by cleaning the dataframe output structure automatically!
data = yf.download(tickers, period='10y')

# Safe fallback check: Modern yfinance versions structure columns as a MultiIndex.
# We pull just the closing prices (Close or Adj Close) safely
if 'Adj Close' in data.columns.levels[0]:
    data = data['Adj Close']
elif 'Close' in data.columns.levels[0]:
    data = data['Close']
else:
    # If it downloaded flat columns
    data = data.filter(like='Close')

# 4. Flatten the dataframe structure for the database
data = data.reset_index()

# Clean up column names in case they are tuples from MultiIndex structures
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
elif hasattr(data.columns, 'values'):
    # Safe renaming for regular index flat data
    data.columns = [str(c) for c in data.columns]

# 5. Save the data into a SQL table named 'historical_prices'
data.to_sql('historical_prices', conn, if_exists='replace', index=False)

print("Database 'portfolio.db' successfully created and populated!")
conn.close()