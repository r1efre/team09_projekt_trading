import os
import pandas as pd
import yaml
import joblib

# Laden der Projektparameter aus der Konfigurationsdatei
params = yaml.safe_load(open("../../../conf/params.yaml", "r", encoding="utf-8"))

# Input: vorbereiteter Datensatz für das aktuelle Backtesting
processed_recent = params["BACKTESTING_RECENT"]["PROCESSED_PATH_RECENT"]
data_complete_path = f"{processed_recent}/dataComplete_recent.parquet"

# Output: Zielordner für die skalierten Backtesting-Daten
scaled_recent = params["BACKTESTING_RECENT"]["SCALED_PATH_RECENT"]
os.makedirs(scaled_recent, exist_ok=True)

# Laden des auf historischen Daten trainierten Scalers
scaler_path = "../../../scaler/scaler.joblib"
scaler = joblib.load(scaler_path)

# Laden des aktuellen dataComplete-Datensatzes
df = pd.read_parquet(data_complete_path).copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Spalten, die bei der Feature Selection entfernt wurden
columns_to_drop = ["open", "high", "low", "vwap", "eth_close"]

# Feature-Liste, mit der der Scaler trainiert wurde (18 Features)
if not hasattr(scaler, "feature_names_in_"):
    raise AttributeError("У scaler нет feature_names_in_.")
scaler_features = list(scaler.feature_names_in_)

# Skalierung aller Features (wie beim Training)
X_all = df[scaler_features].copy()
X_all_scaled = pd.DataFrame(scaler.transform(X_all), columns=scaler_features)

# Entfernen der nicht selektierten Features
X = X_all.drop(columns=columns_to_drop, errors="ignore")
X_scaled = X_all_scaled.drop(columns=columns_to_drop, errors="ignore")

# Erzeugen der Zielvariable für das Backtesting
y_test = df[["trend"]].copy()

# Index-Mapping: Zuordnung von Zeile zu Zeitstempel
x_test_index_map = pd.DataFrame({
    "row_id": range(len(df)),
    "timestamp": df["timestamp"].values
})

# Einheitliche Indizes für alle Ausgabedaten
X.index = range(len(df))
X_scaled.index = range(len(df))
y_test.index = range(len(df))

# Speichern der finalen Backtesting-Dateien
X.to_parquet(f"{scaled_recent}/x_test.parquet", index=False)
X_scaled.to_parquet(f"{scaled_recent}/x_test_scaled.parquet", index=False)
y_test.to_parquet(f"{scaled_recent}/y_test.parquet", index=False)
x_test_index_map.to_parquet(f"{scaled_recent}/x_test_index_map.parquet", index=False)

