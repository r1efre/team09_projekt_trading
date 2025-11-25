import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yaml

params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH = params["DATA_ACQUISITION"]["DATA_PATH"]
file_path = f"{DATA_PATH}/dataMerged.parquet"

#
df = pd.read_parquet(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["close"], label="BTC Close", linewidth=1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

ax.xaxis.set_minor_locator(mdates.MonthLocator())

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title("Veränderung der BTC-Close-Preise über den gesamten Betrachtungszeitraum (01/01/2021-01/11/2025)")
ax.set_xlabel("Zeit")
ax.set_ylabel("Close Preis (BTC)")
ax.grid(True, which="major", linestyle="--", alpha=0.7)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend()
fig.tight_layout()
ax.set_xlim(df.index.min(), df.index.max())
plt.savefig("../../images/02_btc_close_2021-2025.png")
plt.close()


#
df = pd.read_parquet(file_path)
numeric_df = df.select_dtypes(include=["int64", "float64"])

corr = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Korrelationsmatrix der Features")
plt.tight_layout()
plt.savefig("../../images/02_heatmap_features.png")
plt.close()

#
df = pd.read_parquet(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

end = df.index.max()
start = end - pd.Timedelta(days=1)

one_day_btc = df.loc[start:end, "close"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(one_day_btc.index, one_day_btc.values, label="BTC Close", linewidth=1.5)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

ax.set_title("Veränderung der BTC-Close-Preise über den Tag")
ax.set_xlabel("Zeit")
ax.set_ylabel("Close Preis (BTC)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("../../images/02_btc_close_day.png")
plt.close()

#
df = pd.read_parquet(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

end = df.index.max()
start = end - pd.Timedelta(days=1)

one_day_eth = df.loc[start:end, "eth_close"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(one_day_eth.index, one_day_eth.values, label="ETH Close", color="orange", linewidth=1.5)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

ax.set_title("Veränderung der ETH-Close-Preise über den Tag")
ax.set_xlabel("Zeit")
ax.set_ylabel("Close Preis (ETH)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("../../images/02_eth_close_day.png")
plt.close()


#
df = pd.read_parquet(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")


end = df.index.max()
start = end - pd.Timedelta(days=30)
df_month = df.loc[start:end]

btc = df_month["close"]
eth = df_month["eth_close"]


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

ax.text(
    0.05, 0.95,
    f"Pearson r = {corr:.2f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.savefig("../../images/02_btc+eth_correlation.png")
plt.close()