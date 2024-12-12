import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import yfinance as yf

ticker_symbol = 'AAPL'
stock_data = yf.Ticker(ticker_symbol)
historical_data = stock_data.history(
    period="1y",   # Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
    interval="1d"  # Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
)
prices = historical_data["Open"]
smooth = prices.rolling(window=10).mean()
valid_indices = smooth.dropna().index
prices = prices.loc[valid_indices]
smooth = smooth.loc[valid_indices]

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
mmscaler = MinMaxScaler()
raw_mmscaled = mmscaler.fit_transform(prices.to_frame())
smooth_mmscaled = mmscaler.fit_transform(smooth.to_frame())

plt.plot(raw_mmscaled, label='Original Prices (MinMax)', color='green', linestyle='--')
plt.plot(smooth_mmscaled, label='Smoothed Prices (MinMax)', color='red', linewidth=2)
plt.title('Price Comparison: MinMax Scaling', fontsize=12)
plt.xlabel('Time', fontsize=10)
plt.ylabel('Price', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
zscaler = StandardScaler()
raw_zscaled = zscaler.fit_transform(prices.to_frame())
smooth_zscaled = zscaler.fit_transform(smooth.to_frame())

plt.plot(raw_zscaled, label='Original Prices (Z-score)', color='green', linestyle='--')
plt.plot(smooth_zscaled, label='Smoothed Prices (Z-score)', color='red', linewidth=2)
plt.title('Price Comparison: Z-score Normalization', fontsize=12)
plt.xlabel('Time', fontsize=10)
plt.ylabel('Price', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()