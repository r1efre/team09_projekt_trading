import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yaml

# Load API credentials from YAML configuration file
params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH = params["DATA_ACQUISITION"]["DATA_PATH"]
file_path = f"{DATA_PATH}/dataMerged.parquet"

# Load data, convert timestamp column to datetime and sort by timestamp
df = pd.read_parquet(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

# Describe data

print(df.dtypes)

print(df.isna().sum())

desc = df.describe()
print(df.describe().to_string())



# 1) Plot the closing price of BTC over time
# Was run twice, with data for different time windows
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(df.index, df["close"], label="BTC Close", linewidth=1)

# Set major and minor ticks
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.set_title(
    "Veränderung der BTC-Close-Preise über den gesamten Betrachtungszeitraum "
    "(01/01/2021-01/11/2025)"
)
ax.set_xlabel("Zeit")
ax.set_ylabel("Close Preis (BTC)")

ax.grid(True, which="major", linestyle="--", alpha=0.7)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()

# Set axis limits
ax.set_xlim(df.index.min(), df.index.max())

fig.tight_layout()
plt.savefig("../../images/02_btc_close_2021-2025.png")
plt.close()



# 2) BTC OHLCV average daily trend (based on last year)

end = df.index.max()
start = end - pd.Timedelta(days=365)
df_year = df.loc[start:end].copy()

# Extract hour of day
df_year["hour"] = df_year.index.hour

# Average OHLCV per hour over the last year
hourly_ohlcv = df_year.groupby("hour")[["open", "high", "low", "close", "volume"]].mean()

fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot OHLC lines over 24 hours
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["open"],  label="Open",  linewidth=1)
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["high"],  label="High",  linewidth=1)
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["low"],   label="Low",   linewidth=1)
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["close"], label="Close", linewidth=1.3)

ax1.set_xlabel("Stunde des Tages")
ax1.set_ylabel("Preis (BTC)")
ax1.set_title("Durchschnittlicher BTC-OHLC-Verlauf und Volumen pro Stunde (auf Basis des letzten Jahres)")

ax1.set_xticks(range(0, 24))
ax1.set_xticklabels([f"{h:02d}:00" for h in range(24)])
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Volume on secondary axis
ax2 = ax1.twinx()
ax2.bar(
    hourly_ohlcv.index,
    hourly_ohlcv["volume"],
    alpha=0.2,
    color="grey",
    label="Volume"
)
ax2.set_ylabel("Durchschnittliches Trading-Volumen")

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.tight_layout()
plt.savefig("../../images/02_btc_ohlcv_mean_intraday.png")
plt.close()



# 3) Average BTC high-low range per hour (based on last year)

eend = df.index.max()
start = end - pd.Timedelta(days=365)
df_year = df.loc[start:end].copy()

# Hourly range as volatility measure
df_year["range"] = df_year["high"] - df_year["low"]
df_year["hour"] = df_year.index.hour

# Average volatility per hour
hourly_range = df_year.groupby("hour")["range"].mean()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    hourly_range.index,
    hourly_range.values,
    marker="o",
    linewidth=2,
)

ax.set_title("Durchschnittlicher stündlicher High–Low-Range von BTC (auf Basis des letzten Jahres)")
ax.set_xlabel("Stunde des Tages")
ax.set_ylabel("Durchschnittlicher High–Low-Range (BTC)")


ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")

ax.grid(True, linestyle="--", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("../../images/02_btc_high-low_range.png")
plt.close()



# 4) BTC and ETH intraday close prices for the last day

end = df.index.max()
start = end - pd.Timedelta(days=30)
df_30d = df.loc[start:end].copy()

# Extract hour of day
df_30d["hour"] = df_30d.index.hour

# Average close per hour of day for BTC and ETH
hourly_btc = df_30d.groupby("hour")["close"].mean()
hourly_eth = df_30d.groupby("hour")["eth_close"].mean()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()  # second y-axis for ETH

# BTC on left y-axis
line1 = ax1.plot(
    hourly_btc.index,
    hourly_btc.values,
    marker="o",
    linewidth=2,
    color="tab:blue",
    label="BTC Close (avg)"
)

# ETH on right y-axis
line2 = ax2.plot(
    hourly_eth.index,
    hourly_eth.values,
    marker="s",
    linewidth=2,
    color="tab:orange",
    label="ETH Close (avg)"
)

ax1.set_title("Durchschnittliche stündliche BTC- und ETH-Close-Preise (letzte 30 Tage)")
ax1.set_xlabel("Stunde des Tages")
ax1.set_ylabel("BTC Close Preis")
ax2.set_ylabel("ETH Close Preis")

ax1.set_xticks(range(0, 24))
ax1.grid(True, linestyle="--", alpha=0.6)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.tight_layout()
plt.savefig("../../images/02_btc_eth_close_together.png")
plt.close()



# 5) Correlation between BTC and ETH (last 30 days)

end = df.index.max()
start = end - pd.Timedelta(days=30)
df_month = df.loc[start:end]

btc = df_month["close"]
eth = df_month["eth_close"]

# Linear fit for visualizing correlation
corr = btc.corr(eth)
slope, intercept = np.polyfit(btc, eth, 1)
x_line = np.linspace(btc.min(), btc.max(), 200)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(btc, eth, color="purple", alpha=0.6, s=18, edgecolor="none")
ax.plot(x_line, y_line, color="black", linewidth=1)

ax.set_xlabel("BTC Close Preis")
ax.set_ylabel("ETH Close Preis")
ax.set_title("Korrelation zwischen BTC und ETH (letzte 30 Tage)")
ax.grid(True, linestyle="--", alpha=0.5)

# Add text box with correlation value (Pearson r)
ax.text(
    0.05, 0.95,
    f"Pearson r = {corr:.2f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.savefig("../../images/02_btc_eth_correlation.png")
plt.close()



