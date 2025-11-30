import matplotlib.pyplot as plt
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
label_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
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


# 5) Correlation between BTC and ETH

btc = data["btc_return_1h"]
eth = data["eth_return_1h"]

# Linear fit for visualizing correlation
corr = btc.corr(eth)
slope, intercept = np.polyfit(btc, eth, 1)
x_line = np.linspace(btc.min(), btc.max(), 200)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(btc, eth, color="purple", alpha=0.6, s=18, edgecolor="none")
ax.plot(x_line, y_line, color="black", linewidth=1)

ax.set_xlabel("BTC Return 1h")
ax.set_ylabel("ETH Return 1h")
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


#Korrelation BTC Return und BTC Return um eine Stunde zeitversetzt
df = data.set_index('timestamp')
# Rolling Window (168 Stunden = 1 Woche)
window_size = 168

# Berechne ROLLING Correlations mit Zeitverschiebung
correlation_lag1 = df['btc_return_1h'].shift(1).rolling(window=window_size).corr(df['btc_return_1h'])

# Visualisierung
fig, ax = plt.subplots(figsize=(15, 6))

# Plotte die Rolling Correlations
ax.plot(df.index, correlation_lag1, label='Korrelation BTC und BTC zeitversetzt um 1h',
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

#Zeitreihenplot ATR_24 (2021–2025)

plt.figure(figsize=(14, 5))
plt.plot(data["timestamp"], data["atr_24"], label="ATR 24h (in %)")

plt.title("Zeitreihe: ATR (24h) über den gesamten Zeitraum 2021–2025")
plt.xlabel("Datum")
plt.ylabel("ATR 24 (prozentuale Volatilität)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()



# 1) Index zurück in Spalte + Zeitraum wählen (letzte 12 Monate)
df = df.reset_index()  # nur falls index = timestamp war
end = df["timestamp"].max()
start = end - pd.Timedelta(days=365)

df_period = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

# 2) EMA-Differenz berechnen
df_period["ema_diff"] = df_period["ema_6"] - df_period["ema_24"]

# 3) Glättung definieren (z.B. 24h oder 72h)
smooth_window = 100

df_period["ema_diff_smooth"] = df_period["ema_diff"].rolling(window=smooth_window).mean()
df_period["btc_return_1h_smooth"] = df_period["btc_return_1h"].rolling(window=smooth_window).mean()

# 4) Plot vorbereiten
fig, ax1 = plt.subplots(figsize=(14, 6))

# Geglättete EMA-Differenz
ax1.plot(
    df_period["timestamp"],
    df_period["ema_diff_smooth"],
    label=f"EMA6 - EMA24 (geglättet, {smooth_window}h)",
    color="blue", linewidth=1.6
)
ax1.set_xlabel("Datum")
ax1.set_ylabel("EMA6 - EMA24 (smooth)", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle="--", alpha=0.3)

# 5) Zweite Y-Achse für geglätteten BTC Return
ax2 = ax1.twinx()
ax2.plot(
    df_period["timestamp"],
    df_period["btc_return_1h_smooth"],
    label=f"BTC Return 1h (geglättet, {smooth_window}h)",
    color="red", linewidth=1.2,
    alpha=0.7
)
ax2.set_ylabel("BTC Return 1h (smooth)", color="red")
ax2.tick_params(axis='y', labelcolor='red')

# 6) Titel & kombinierte Legende
plt.title(f"EMA6 - EMA24 vs. BTC 1h Return (geglättet, letzte 12 Monate)", fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.show()











