import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = params['DATA_PREP']['PROCESSED_PATH']
data = pd.read_parquet(f'{processed_path}/dataComplete.parquet')

#Barplot der Trendklassen
# Klassen zählen
class_counts = data["trend"].value_counts().sort_index()
label_mapping = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
labels = [label_mapping[i] for i in class_counts.index]
plt.figure(figsize=(6, 4))
plt.bar(labels, class_counts.values)

# Labels
plt.title("Verteilung der Trend-Klassen")
plt.xlabel("Trend-Klasse")
plt.ylabel("Anzahl")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()
print((class_counts / class_counts.sum() * 100).round(2))


#BTC vs ETH – Durchschnittlicher 1h Return pro Tagesstunde (letzte 90 Tage)

# Letzte 90 Tage filtern
last_timestamp = data["timestamp"].max()
start_timestamp = last_timestamp - pd.Timedelta(days=90)

df_90 = data[data["timestamp"] >= start_timestamp].copy()
df_90["hour"] = df_90["timestamp"].dt.hour

# Mittelwerte pro Stunde
hourly_means = df_90.groupby("hour")[["btc_return_1h", "eth_return_1h"]].mean()

# Glättung
hourly_smoothed = hourly_means.rolling(3, center=True).mean()

plt.figure(figsize=(10, 5))

plt.plot(hourly_smoothed.index, hourly_smoothed["btc_return_1h"], label="BTC 1h Return (smooth)")
plt.plot(hourly_smoothed.index, hourly_smoothed["eth_return_1h"], label="ETH 1h Return (smooth)")

plt.title("Durchschnittlicher Return_1h pro Tagesstunde (geglättet, letzte 90 Tage)")
plt.xlabel("Tagesstunde (0–23)")
plt.ylabel("Durchschnittlicher 1h Return")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.xticks(range(0, 24))

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

#Trend-Verteilung alle 2 Monate

# Trend text labels
label_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
data["trend_label"] = data["trend"].map(label_map)

# --- 2-Monats-Periode erzeugen ---
def two_month_period(ts):
    year = ts.year
    month = ts.month
    # block: 1-2, 3-4, 5-6, ...
    block_start = month - (month - 1) % 2
    block_end = block_start + 1
    return f"{year}-{block_start:02d}-{block_end:02d}"

data["period_2m"] = data["timestamp"].apply(two_month_period)

# --- Gruppieren & zählen ---
trend_counts = data.groupby(["period_2m", "trend_label"]).size().unstack(fill_value=0)

# --- Plot ---
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






