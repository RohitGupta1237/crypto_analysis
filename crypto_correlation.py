import pandas_datareader as web
import mplfinance as mpf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

currency = "USD"
metric = "Close"
start = dt.datetime(2013, 4, 29)
end = dt.datetime.now()

crypto = ['BTC', 'ETH', 'ADA', 'SOL', 'LTC', 'TRX', 'XRP', 'MATIC', 'XMR', 'SIN']
colnames = []

first = True

# Fetch data for all cryptocurrencies
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
