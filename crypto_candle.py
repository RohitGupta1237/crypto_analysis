import pandas_datareader as web
import datetime as dt
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

# Function to calculate moving averages
def calculate_moving_averages(data, short_window=50, long_window=200):
    data['50_MA'] = data['Close'].rolling(window=short_window).mean()
    data['200_MA'] = data['Close'].rolling(window=long_window).mean()
    return data

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std)
    return data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['MACD'] = data['Close'].ewm(span=short_window, adjust=False).mean() - data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Set the time range
start = dt.datetime(2013, 4, 29)
end = dt.datetime.now()

# Fetch data from Yahoo Finance
data = web.DataReader("BTC-USD", "yahoo", start, end)

# Calculate moving averages, Bollinger Bands, RSI, and MACD
data = calculate_moving_averages(data)
data = calculate_bollinger_bands(data)
data = calculate_rsi(data)
data = calculate_macd(data)

# Plot the candlestick chart with Moving Averages and Bollinger Bands
add_plots = [
    mpf.make_addplot(data['50_MA'], color='blue', width=0.8),
    mpf.make_addplot(data['200_MA'], color='green', width=0.8),
    mpf.make_addplot(data['Bollinger_Upper'], color='orange', linestyle='--'),
    mpf.make_addplot(data['Bollinger_Lower'], color='orange', linestyle='--'),
    mpf.make_addplot(data['RSI'], panel=1, color='purple', ylabel='RSI'),
    mpf.make_addplot(data['MACD'], panel=2, color='red', ylabel='MACD'),
    mpf.make_addplot(data['MACD_Signal'], panel=2, color='blue')
]

# Customize the style
style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'axes.grid': True})

# Plot the chart
mpf.plot(
    data,
    type="candle",
    volume=True,
    style=style,
    addplot=add_plots,
    title="BTC-USD with Moving Averages, Bollinger Bands, RSI, and MACD",
    ylabel="Price",
    panel_ratios=(6, 2, 2),  # Adjust the size ratios between price, RSI, and MACD
    figratio=(14, 8),
    figscale=1.2,
)

# Optional: Display moving averages, RSI, and MACD as individual line plots
plt.show()
