import pandas_datareader as web
import datetime as dt
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# Function to calculate Average True Range (ATR)
def calculate_atr(data, window=14):
    data['High-Low'] = data['High'] - data['Low']
    data['High-Prev Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Prev Close'] = abs(data['Low'] - data['Close'].shift(1))
    tr = data[['High-Low', 'High-Prev Close', 'Low-Prev Close']].max(axis=1)
    data['ATR'] = tr.rolling(window=window).mean()
    return data

# Function to calculate Stochastic Oscillator
def calculate_stochastic(data, window=14):
    data['L14'] = data['Low'].rolling(window=window).min()
    data['H14'] = data['High'].rolling(window=window).max()
    data['%K'] = (data['Close'] - data['L14']) / (data['H14'] - data['L14']) * 100
    data['%D'] = data['%K'].rolling(window=3).mean()  # Smooth %K to get %D
    return data

# Function to calculate On-Balance Volume (OBV)
def calculate_obv(data):
    data['OBV'] = 0
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            data['OBV'][i] = data['OBV'][i - 1] + data['Volume'][i]
        elif data['Close'][i] < data['Close'][i - 1]:
            data['OBV'][i] = data['OBV'][i - 1] - data['Volume'][i]
        else:
            data['OBV'][i] = data['OBV'][i - 1]
    return data

# Function to generate buy/sell signals
def generate_signals(data):
    data['Signal'] = 0
    data['Signal'] = np.where((data['RSI'] < 30) & (data['Close'] > data['50_MA']), 1, data['Signal'])  # Buy signal
    data['Signal'] = np.where((data['RSI'] > 70) & (data['Close'] < data['50_MA']), -1, data['Signal'])  # Sell signal
    return data

# Set the time range
start = dt.datetime(2013, 4, 29)
end = dt.datetime.now()

# Fetch data from Yahoo Finance
data = web.DataReader("BTC-USD", "yahoo", start, end)

# Calculate indicators
data = calculate_moving_averages(data)
data = calculate_bollinger_bands(data)
data = calculate_rsi(data)
data = calculate_macd(data)
data = calculate_atr(data)
data = calculate_stochastic(data)
data = calculate_obv(data)
data = generate_signals(data)

# Plot the candlestick chart with Moving Averages and Bollinger Bands
add_plots = [
    mpf.make_addplot(data['50_MA'], color='blue', width=0.8),
    mpf.make_addplot(data['200_MA'], color='green', width=0.8),
    mpf.make_addplot(data['Bollinger_Upper'], color='orange', linestyle='--'),
    mpf.make_addplot(data['Bollinger_Lower'], color='orange', linestyle='--'),
    mpf.make_addplot(data['RSI'], panel=1, color='purple', ylabel='RSI'),
    mpf.make_addplot(data['MACD'], panel=2, color='red', ylabel='MACD'),
    mpf.make_addplot(data['MACD_Signal'], panel=2, color='blue'),
    mpf.make_addplot(data['%K'], panel=3, color='cyan', ylabel='Stochastic %K'),
    mpf.make_addplot(data['%D'], panel=3, color='magenta'),
]

# Customize the style
style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'axes.grid': True})

# Add buy/sell signals to the candlestick plot
buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]

mpf.plot(
    data,
    type="candle",
    volume=True,
    style=style,
    addplot=add_plots,
    title="BTC-USD with Indicators and Trading Signals",
    ylabel="Price",
    panel_ratios=(6, 2, 2, 2),  # Adjust the size ratios between price and indicators
    figratio=(14, 8),
    figscale=1.2,
    buysell=buy_signals[['Close']].assign(marker='^', color='green'),
    buycolor='green', sellcolor='red',
    markersize=8,
)

# Show the chart
plt.show()
