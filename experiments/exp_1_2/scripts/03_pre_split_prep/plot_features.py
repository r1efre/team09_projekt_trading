import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = params['DATA_PREP']['PROCESSED_PATH']
data = pd.read_parquet(f'{processed_path}/dataComplete.parquet')

#Trend-Verteilung alle 2 Monate

# Trend text labels
label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
data["trend_label"] = data["trend"].map(label_map)

# 2-Monats-Periode erzeugen
def two_month_period(ts):
    year = ts.year
    month = ts.month
    block_start = month - (month - 1) % 2
    block_end = block_start + 1
    return f"{year}-{block_start:02d}-{block_end:02d}"

data["period_2m"] = data["timestamp"].apply(two_month_period)

# Gruppieren & zählen
trend_counts = data.groupby(["period_2m", "trend_label"]).size().unstack(fill_value=0)

# Plot
trend_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 6),
    colormap="viridis"
)

plt.title("Trend-Verteilung pro 2-Monats-Periode")
plt.xlabel("2-Monats-Periode")
plt.ylabel("Anzahl der Stunden")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.show()

trend_distribution = (
    data["trend"]
    .value_counts(normalize=True)
    .mul(100)
)

print(trend_distribution)

# Correlation between BTC and ETH

btc = data["btc_return_60min"]
eth = data["eth_return_60min"]

# Pearson-Korrelationskoeffizienten r
corr = btc.corr(eth)
slope, intercept = np.polyfit(btc, eth, 1)
x_line = np.linspace(btc.min(), btc.max(), 200)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(btc, eth, color="purple", alpha=0.6, s=18, edgecolor="none")
ax.plot(x_line, y_line, color="black", linewidth=1)

ax.set_xlabel("BTC Return 60min")
ax.set_ylabel("ETH Return 60min")
ax.set_title("Korrelation zwischen BTC und ETH")
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
plt.show()





# #Korrelation BTC Return und BTC Return um eine Stunde zeitversetzt
df = data.set_index('timestamp')
# Rolling Window (10080 Minuten = 1 Woche)
window_size = 10080

# Berechne Rolling Correlations mit Zeitverschiebung
correlation_lag1 = df['btc_return_60min'].shift(60).rolling(window_size).corr(df['btc_return_60min'])

# Visualisierung
fig, ax = plt.subplots(figsize=(15, 6))

# Plotte die Rolling Correlations
ax.plot(df.index, correlation_lag1, label='Korrelation BTC und BTC zeitversetzt um 60min',
        linewidth=1.5, color='blue', alpha=0.8)

# Referenzlinien
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Keine Korrelation')
ax.axhline(y=0.3, color='green', linestyle=':', linewidth=0.8, alpha=0.5, label='Moderate Korrelation (0.3)')
ax.axhline(y=-0.3, color='green', linestyle=':', linewidth=0.8, alpha=0.5)

# Styling
ax.set_xlabel('Datum', fontsize=12)
ax.set_ylabel('Korrelation', fontsize=12)
ax.set_title(f'Rolling Lag-Correlation: Kann BTC-Vergangenheit die BTC-Entwicklung vorhersagen? (Window: {window_size}h)',
             fontsize=14, fontweight='bold')
ax.set_ylim(-1, 1)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()





# #Zusammenhang zwischen EMA Differenz und BTC_Return

df = df.reset_index()
end = df["timestamp"].max()
start = end - pd.Timedelta(days=30)

df_period = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

#EMA-Differenz berechnen
df_period["ema_diff"] = df_period["ema_20"] - df_period["ema_200"]

# 3) Glättung definieren
smooth_window = 100

#Glättung
df_period["ema_diff_smooth"] = df_period["ema_diff"].rolling(window=smooth_window).mean()
df_period["btc_return_60min_smooth"] = df_period["btc_return_60min"].rolling(window=smooth_window).mean()

fig, ax1 = plt.subplots(figsize=(14, 6))

# Geglättete EMA-Differenz
ax1.plot(
    df_period["timestamp"],
    df_period["ema_diff_smooth"],
    label=f"EMA20 - EMA200 (geglättet, {smooth_window}h)",
    color="blue", linewidth=1.6
)
ax1.set_xlabel("Datum")
ax1.set_ylabel("EMA20 - EMA200 (smooth)", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle="--", alpha=0.3)

#Zweite Y-Achse für geglätteten BTC Return
ax2 = ax1.twinx()
ax2.plot(
    df_period["timestamp"],
    df_period["btc_return_60min_smooth"],
    label=f"BTC Return 60min (geglättet, {smooth_window}h)",
    color="red", linewidth=1.2,
    alpha=0.7
)
ax2.set_ylabel("BTC Return 60min (smooth)", color="red")
ax2.tick_params(axis='y', labelcolor='red')

#Titel & kombinierte Legende
plt.title(f"EMA20 - EMA200 vs. BTC 60min Return (geglättet, letzte 30 Tage)", fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.show()