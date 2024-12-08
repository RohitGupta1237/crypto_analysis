import pandas_datareader as web
import datetime as dt
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Constants for indicators and strategies
SHORT_WINDOW = 50
LONG_WINDOW = 200
BOLLINGER_WINDOW = 20
RSI_WINDOW = 14
MACD_SHORT = 12
MACD_LONG = 26
MACD_SIGNAL = 9
ATR_WINDOW = 14
STOCHASTIC_WINDOW = 14

# Function to calculate moving averages
def calculate_moving_averages(data, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    data['50_MA'] = data['Close'].rolling(window=short_window).mean()
    data['200_MA'] = data['Close'].rolling(window=long_window).mean()
    return data

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=BOLLINGER_WINDOW, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std)
    return data

# Function to calculate RSI
def calculate_rsi(data, window=RSI_WINDOW):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=MACD_SHORT, long_window=MACD_LONG, signal_window=MACD_SIGNAL):
    data['MACD'] = data['Close'].ewm(span=short_window, adjust=False).mean() - data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate Average True Range (ATR)
def calculate_atr(data, window=ATR_WINDOW):
    data['High-Low'] = data['High'] - data['Low']
    data['High-Prev Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Prev Close'] = abs(data['Low'] - data['Close'].shift(1))
    tr = data[['High-Low', 'High-Prev Close', 'Low-Prev Close']].max(axis=1)
    data['ATR'] = tr.rolling(window=window).mean()
    return data

# Function to calculate Stochastic Oscillator
def calculate_stochastic(data, window=STOCHASTIC_WINDOW):
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

# Function to generate buy/sell signals based on a variety of indicators
def generate_signals(data):
    data['Signal'] = 0
    # Buy when RSI is oversold and price crosses above the 50 MA
    data['Signal'] = np.where((data['RSI'] < 30) & (data['Close'] > data['50_MA']), 1, data['Signal'])
    # Sell when RSI is overbought and price crosses below the 50 MA
    data['Signal'] = np.where((data['RSI'] > 70) & (data['Close'] < data['50_MA']), -1, data['Signal'])
    # Add additional conditions (e.g., MACD crossover)
    data['Signal'] = np.where((data['MACD'] > data['MACD_Signal']) & (data['RSI'] < 70), 1, data['Signal'])  # Buy when MACD crosses above signal
    data['Signal'] = np.where((data['MACD'] < data['MACD_Signal']) & (data['RSI'] > 30), -1, data['Signal'])  # Sell when MACD crosses below signal
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

# Plot the candlestick chart with additional indicators
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
    mpf.make_addplot(data['ATR'], panel=4, color='brown', ylabel='ATR'),
    mpf.make_addplot(data['OBV'], panel=5, color='black', ylabel='OBV'),
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
    panel_ratios=(6, 2, 2, 2, 2, 2),  # Adjust the size ratios between price and indicators
    figratio=(14, 8),
    figscale=1.2,
    buysell=buy_signals[['Close']].assign(marker='^', color='green'),
    sellbuy=sell_signals[['Close']].assign(marker='v', color='red'),
    markersize=8,
)

# Show the chart
plt.show()

# Additional feature: Volatility analysis using Bollinger Band Width
data['BB_Width'] = data['Bollinger_Upper'] - data['Bollinger_Lower']
sns.lineplot(data=data, x=data.index, y='BB_Width')
plt.title("Bollinger Band Width Analysis")
plt.xlabel("Date")
plt.ylabel("Bollinger Band Width")
plt.show()

# Additional feature: Correlation between OBV and price
sns.scatterplot(x=data['OBV'], y=data['Close'])
plt.title("OBV vs Price")
plt.xlabel("On-Balance Volume")
plt.ylabel("Price")
plt.show()

# Calculate and plot the correlation matrix of selected features
correlation_data = data[['Close', '50_MA', '200_MA', 'RSI', 'MACD', 'ATR', 'OBV']]
correlation_matrix = correlation_data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Indicators")
plt.show()
