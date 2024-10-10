import pandas_datareader as web
import mplfinance as mpf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Define constants
currency = "USD"
metric = "Close"
start = dt.datetime(2013, 4, 29)
end = dt.datetime.now()

crypto = ['BTC', 'ETH', 'ADA', 'SOL', 'LTC', 'TRX', 'XRP', 'MATIC', 'XMR', 'SIN']
colnames = []

# Fetch data for all cryptocurrencies
first = True
for ticker in crypto:
    data = web.DataReader(f"{ticker}-{currency}", "yahoo", start, end)
    if first:
        combined = data[[metric]].copy()
        colnames.append(ticker)
        combined.columns = colnames
        first = False
    else:
        combined = combined.join(data[metric])
        colnames.append(ticker)
        combined.columns = colnames

# Plot Normalized Data
plt.figure(figsize=(12, 8))
plt.yscale('log')  # Set log scale for better comparison across cryptos
for ticker in crypto:
    normalized = combined[ticker] / combined[ticker].iloc[0]  # Normalize data
    plt.plot(normalized, label=ticker)

plt.title('Normalized Cryptocurrency Prices')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend(loc="upper right")
plt.show()

# Add Exponential Moving Averages (EMA)
window_short = 20
window_long = 50

for ticker in crypto:
    combined[f"{ticker}_EMA_{window_short}"] = combined[ticker].ewm(span=window_short, adjust=False).mean()
    combined[f"{ticker}_EMA_{window_long}"] = combined[ticker].ewm(span=window_long, adjust=False).mean()

# Plot with EMA overlays
plt.figure(figsize=(12, 8))
for ticker in crypto:
    plt.plot(combined[ticker], label=f"{ticker} Price")
    plt.plot(combined[f"{ticker}_EMA_{window_short}"], linestyle='--', label=f"{ticker} EMA-{window_short}")
    plt.plot(combined[f"{ticker}_EMA_{window_long}"], linestyle='--', label=f"{ticker} EMA-{window_long}")

plt.yscale('log')  # Keep the log scale for better visualization
plt.title('Cryptocurrency Prices with EMAs')
plt.legend(loc="upper right")
plt.show()

# Calculate Volatility (Standard Deviation) and Plot
volatility_window = 30  # Use a 30-day rolling window for volatility
plt.figure(figsize=(12, 8))
for ticker in crypto:
    combined[f"{ticker}_Volatility"] = combined[ticker].rolling(window=volatility_window).std()
    plt.plot(combined[f"{ticker}_Volatility"], label=f"{ticker} Volatility")

plt.title('Cryptocurrency Volatility (Rolling 30-day Std. Dev.)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend(loc="upper right")
plt.show()

# Volume Analysis
for ticker in crypto:
    data = web.DataReader(f"{ticker}-{currency}", "yahoo", start, end)
    combined[f"{ticker}_Volume_Avg"] = data['Volume'].rolling(window=30).mean()

# Plot Volume with Average Volume
plt.figure(figsize=(12, 8))
for ticker in crypto:
    plt.plot(data['Volume'], label=f"{ticker} Volume", alpha=0.3)
    plt.plot(combined[f"{ticker}_Volume_Avg"], label=f"{ticker} Avg Volume", linestyle='--')

plt.title('Cryptocurrency Volume Analysis')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend(loc="upper right")
plt.show()

# Sharpe Ratio Calculation
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

sharpe_ratios = {}
for ticker in crypto:
    returns = combined[ticker].pct_change().dropna()
    sharpe_ratios[ticker] = calculate_sharpe_ratio(returns)

print("Sharpe Ratios:")
print(sharpe_ratios)

# Maximum Drawdown Calculation
def calculate_max_drawdown(prices):
    drawdowns = (prices / prices.cummax()) - 1
    return drawdowns.min()

max_drawdowns = {}
for ticker in crypto:
    max_drawdowns[ticker] = calculate_max_drawdown(combined[ticker])

print("Maximum Drawdowns:")
print(max_drawdowns)

# Seasonality Analysis
combined['Month'] = combined.index.month
combined['Weekday'] = combined.index.weekday

# Average Monthly Returns
monthly_returns = combined.groupby('Month').mean().pct_change()

# Plot Monthly Returns
plt.figure(figsize=(12, 6))
monthly_returns.plot(kind='bar', color='skyblue')
plt.title('Average Monthly Returns')
plt.xlabel('Month')
plt.ylabel('Average Return')
plt.show()

# Average Weekly Returns
weekly_returns = combined.groupby('Weekday').mean().pct_change()

# Plot Weekly Returns
plt.figure(figsize=(12, 6))
weekly_returns.plot(kind='bar', color='salmon')
plt.title('Average Weekly Returns')
plt.xlabel('Weekday (0=Monday)')
plt.ylabel('Average Return')
plt.show()

# Correlation Heatmap with Additional Correlation Methods
combined_pct_change = combined.pct_change()  # Calculate daily returns
correlation_matrix_pearson = combined_pct_change.corr(method='pearson')
correlation_matrix_kendall = combined_pct_change.corr(method='kendall')
correlation_matrix_spearman = combined_pct_change.corr(method='spearman')

# Customize Heatmaps: Pearson, Kendall, Spearman
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Pearson
sns.heatmap(correlation_matrix_pearson, annot=True, cmap="coolwarm", ax=ax[0])
ax[0].set_title('Pearson Correlation')

# Kendall
sns.heatmap(correlation_matrix_kendall, annot=True, cmap="coolwarm", ax=ax[1])
ax[1].set_title('Kendall Correlation')

# Spearman
sns.heatmap(correlation_matrix_spearman, annot=True, cmap="coolwarm", ax=ax[2])
ax[2].set_title('Spearman Correlation')

plt.show()

# Pairplot of Cryptos (for a subset)
subset_crypto = ['BTC', 'ETH', 'ADA', 'XRP']  # Smaller subset to avoid overplotting
subset_data = combined[subset_crypto].pct_change().dropna()

# Plot Pairplot using Seaborn
sns.pairplot(subset_data)
plt.show()

# Optional: Mask Upper Triangle of the Correlation Heatmap
mask = np.triu(np.ones_like(correlation_matrix_pearson, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_pearson, annot=True, cmap="coolwarm", mask=mask)
plt.title('Pearson Correlation (Masked Upper Triangle)')
plt.show()

# Advanced Visualization with Plotly
# Plotting interactive candlestick chart for BTC
btc_data = web.DataReader("BTC-USD", "yahoo", start, end)

fig = go.Figure(data=[go.Candlestick(x=btc_data.index,
                                       open=btc_data['Open'],
                                       high=btc_data['High'],
                                       low=btc_data['Low'],
                                       close=btc_data['Close'])])

fig.update_layout(title='BTC-USD Candlestick Chart', xaxis_title='Date', yaxis_title='Price (USD)')
fig.show()

# Simple Machine Learning Model for Price Prediction
# Prepare data
X = np.arange(len(combined)).reshape(-1, 1)  # Use index as feature
y = combined['BTC'].values  # Target variable (BTC prices)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(combined['BTC'], label='Actual BTC Prices')
plt.plot(range(len(X_train), len(X_train) + len(predictions)), predictions, label='Predicted BTC Prices', linestyle='--')
plt.title('BTC Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
