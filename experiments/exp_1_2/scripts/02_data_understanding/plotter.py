import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from matplotlib.ticker import MultipleLocator

# Load API credentials from YAML configuration file
params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = params["DATA_ACQUISITION"]["DATA_PATH"]
raw_data_file = params['DATA_PREP']['RAW_DATA_FILE']
file_path = f"{data_path}/{raw_data_file}"

# Load data, convert timestamp column to datetime and sort by timestamp
df = pd.read_parquet(file_path)

# Describe data

print(df.dtypes)

print(df.isna().sum())

desc = df.describe()
print(df.describe().to_string())



# 1) Plot the closing price of BTC over time
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(df.index, df["close"], label="BTC Close")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.set_title(
    "Veränderung der BTC-Close-Preise über den gesamten Betrachtungszeitraum "
    "(01/01/2024-30/11/2025)"
)
ax.set_xlabel("Zeit")
ax.set_ylabel("Close Preis (BTC)")

ax.grid(True, which="major", linestyle="--", alpha=0.7)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()

ax.set_xlim(df.index.min(), df.index.max())

fig.tight_layout()
plt.savefig("../../images/02_btc_close_2024-2025.png")
plt.close()

# Zeitraum explizit Oktober 2025
start = "2025-10-01 00:00:00"
end   = "2025-10-31 23:59:59"
df_oct = df.loc[start:end]

fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

ax.plot(
    df_oct.index,
    df_oct["close"],
    label="BTC Close (Minuten)",
    linewidth=0.7
)

# 🔹 JEDER TAG als Tick
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.set_title("Veränderung der BTC-Schlusskurse im Oktober 2025 (01.10.2025–31.10.2025)")
ax.set_xlabel("Zeit")
ax.set_ylabel("Close Preis (BTC)")

ax.grid(True, linestyle="--", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()

ax.set_xlim(df_oct.index.min(), df_oct.index.max())

plt.savefig("../../images/02_btc_close_minutes_oct_2025.png")
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

# 2) BTC OHLCV average daily trend (based on last year)

end = df.index.max()
start = end - pd.Timedelta(days=365)
df_year = df.loc[start:end].copy()

# Extract hour of day
df_year["hour"] = df_year.index.hour

# Average OHLCV per hour over the last year
hourly_ohlcv = df_year.groupby("hour")[["open", "high", "low", "close", "volume"]].mean()

fig, ax1 = plt.subplots(figsize=(18, 6))  # ← breiter für 48 Labels

# Plot OHLC lines over 24 hours
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["open"],  label="Open",  linewidth=1)
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["high"],  label="High",  linewidth=1)
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["low"],   label="Low",   linewidth=1)
ax1.plot(hourly_ohlcv.index, hourly_ohlcv["close"], label="Close", linewidth=1.3)

ax1.set_xlabel("Stunde des Tages")
ax1.set_ylabel("Preis (BTC)")
ax1.set_title("Durchschnittlicher BTC-OHLC-Verlauf und Volumen pro Stunde (auf Basis des letzten Jahres)")

#  30-Minuten-Beschriftung (00:00, 00:30, 01:00, ...)
ticks_30m = [h + m for h in range(24) for m in (0.0, 0.5)]
labels_30m = [f"{int(t):02d}:{'00' if t % 1 == 0 else '30'}" for t in ticks_30m]

ax1.set_xticks(ticks_30m)
ax1.set_xticklabels(labels_30m, rotation=45, ha="right", fontsize=8)

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

# Mehr Platz unten für die 45° Labels
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

plt.savefig("../../images/02_btc_ohlcv_mean_intraday.30.png")
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

# 3.1
# Average BTC 60-minute range per hour (based on last year)
# -> aligned with target horizon (t -> t+60)

end = df.index.max()
start = end - pd.Timedelta(days=365)
df_year = df.loc[start:end].copy()

# Stunde des Tages
df_year["hour"] = df_year.index.hour

# 60-Minuten-Range (rollend, auf Minutendaten)
df_year["range_60m"] = (
    df_year["high"].rolling(window=60).max() -
    df_year["low"].rolling(window=60).min()
)

# Durchschnittliche 60-Minuten-Range pro Stunde
hourly_range_60m = df_year.groupby("hour")["range_60m"].mean()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    hourly_range_60m.index,
    hourly_range_60m.values,
    marker="o",
    linewidth=2,
)

ax.set_title(
    "Durchschnittliche 60-Minuten-Range von BTC nach Stunde (letztes Jahr)"
)
ax.set_xlabel("Stunde des Tages")
ax.set_ylabel("BTC-Preisbewegung")

ax.set_xticks(range(24))
ax.set_xticklabels(range(24))  # 0–23 ist hier am klarsten

ax.grid(True, linestyle="--", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("../../images/02_btc_range_60m_by_hour.png")
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

# 4.1)
end = df.index.max()
start = end - pd.Timedelta(days=7)
df_7d = df.loc[start:end].copy()

# 30-Minuten-Slot pro Tag (0..47)
# 00:00 -> 0, 00:30 -> 1, 01:00 -> 2, ...
df_7d["slot_30m"] = df_7d.index.hour * 2 + (df_7d.index.minute // 30)

# Durchschnittlicher Close pro 30-Minuten-Slot (über 7 Tage)
btc_30m = df_7d.groupby("slot_30m")["close"].mean().reindex(range(48))
eth_30m = df_7d.groupby("slot_30m")["eth_close"].mean().reindex(range(48))

# X-Achse als Stunde des Tages (0.0, 0.5, 1.0, ..., 23.5)
x = btc_30m.index / 2.0

# Plot
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

# BTC (linke Achse)
line1 = ax1.plot(
    x,
    btc_30m.values,
    marker="o",
    linewidth=2,
    color="tab:blue",
    label="BTC Close (avg)"
)

# ETH (rechte Achse)
line2 = ax2.plot(
    x,
    eth_30m.values,
    marker="s",
    linewidth=2,
    color="tab:orange",
    label="ETH Close (avg)"
)

# Titel & Labels
ax1.set_title("Durchschnittliche 30-minütliche BTC- und ETH-Close-Preise (letzte 7 Tage)")
ax1.set_xlabel("Stunde des Tages (0–23)")
ax1.set_ylabel("BTC Close Preis")
ax2.set_ylabel("ETH Close Preis")

# X-Achse: nur Stunden (numerisch)
ax1.set_xlim(0, 23.5)
ax1.set_xticks(range(0, 24))
ax1.set_xticklabels(range(0, 24))

# 30-Minuten-Raster als feines Grid (ohne zusätzliche Labels)
ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax1.grid(True, which="major", linestyle="--", alpha=0.6)
ax1.grid(True, which="minor", linestyle=":", alpha=0.25)

# Legende kombinieren
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.tight_layout()
plt.savefig("../../images/02_btc_eth_close_7d_30min.png")
plt.close()


# 4.2
end = df.index.max()
start = end - pd.Timedelta(days=14)
df_14d = df.loc[start:end].copy()

# 30-Minuten-Slot pro Tag (0..47)
df_14d["slot_30m"] = df_14d.index.hour * 2 + (df_14d.index.minute // 30)

# Durchschnittlicher Close pro 30-Minuten-Slot (über 14 Tage)
btc_30m = df_14d.groupby("slot_30m")["close"].mean().reindex(range(48))
eth_30m = df_14d.groupby("slot_30m")["eth_close"].mean().reindex(range(48))

# X-Achse als Stunde des Tages (0.0, 0.5, 1.0, ..., 23.5)
x = btc_30m.index / 2.0

fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

# BTC (linke Achse)
line1 = ax1.plot(
    x,
    btc_30m.values,
    marker="o",
    linewidth=2,
    color="tab:blue",
    label="BTC Close (avg)"
)

# ETH (rechte Achse)
line2 = ax2.plot(
    x,
    eth_30m.values,
    marker="s",
    linewidth=2,
    color="tab:orange",
    label="ETH Close (avg)"
)

ax1.set_title("Durchschnittliche 30-minütliche BTC- und ETH-Close-Preise (letzte 14 Tage)")
ax1.set_xlabel("Stunde des Tages (0–23)")
ax1.set_ylabel("BTC Close Preis")
ax2.set_ylabel("ETH Close Preis")

ax1.set_xlim(0, 23.5)
ax1.set_xticks(range(0, 24))
ax1.set_xticklabels(range(0, 24))

ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax1.grid(True, which="major", linestyle="--", alpha=0.6)
ax1.grid(True, which="minor", linestyle=":", alpha=0.25)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.tight_layout()
plt.savefig("../../images/02_btc_eth_close_14d_30min.png")
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

# 6. NEW: Intraday distribution of 1-minute returns (Data Understanding)
import matplotlib.patches as mpatches

end = df.index.max()
start = end - pd.Timedelta(days=365)
df_id = df.loc[start:end].copy()

df_id["ret_1m"] = df_id["close"].pct_change()
df_id["hour"] = df_id.index.hour

# Daten je Stunde sammeln
data = [df_id[df_id["hour"] == h]["ret_1m"].dropna().values for h in range(24)]

# "Aktivität" je Stunde über Streuung bestimmen
stds = [np.std(vals) for vals in data]
q1, q2 = np.quantile(stds, [0.33, 0.66])  # automatisch: ruhig / normal / aktiv

def hour_color(std):
    if std <= q1:
        return "tab:blue"     # ruhig
    elif std <= q2:
        return "grey"         # normal
    return "tab:orange"       # sehr aktiv

colors = [hour_color(s) for s in stds]

fig, ax = plt.subplots(figsize=(16, 6))

bp = ax.boxplot(
    data,
    positions=range(24),
    widths=0.6,
    showfliers=False,
    patch_artist=True
)

# Farben setzen (minimal)
for box, col in zip(bp["boxes"], colors):
    box.set_facecolor(col)
    box.set_alpha(0.35)

ax.set_title("Bitcoin: Preisänderung pro Minute je Uhrzeit")
ax.set_xlabel("Stunde des Tages")
ax.set_ylabel("Preisänderung pro Minute")

ax.set_xticks(range(24))
ax.set_xticklabels(range(24))

ax.grid(True, linestyle="--", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legende (einfach)
ax.legend(
    handles=[
        mpatches.Patch(color="tab:blue", alpha=0.35, label="ruhig"),
        mpatches.Patch(color="grey", alpha=0.35, label="normal"),
        mpatches.Patch(color="tab:orange", alpha=0.35, label="sehr aktiv"),
    ],
    loc="upper left"
)

plt.tight_layout()
plt.savefig("../../images/02_intraday_minute_change.png", dpi=200)
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