import os
import pandas as pd
import yaml
import joblib

# Laden der Projektparameter aus der Konfigurationsdatei
params = yaml.safe_load(open("../../../conf/params.yaml", "r", encoding="utf-8"))

# Input: vollständig vorbereiteter Datensatz
# (aktuelle Daten für das Backtesting)
processed_recent = params["BACKTESTING_RECENT"]["PROCESSED_PATH_RECENT"]
data_complete_path = f"{processed_recent}/dataComplete_recent.parquet"

# Output: Zielordner für die skalierten Testdaten
scaled_recent = params["BACKTESTING_RECENT"]["SCALED_PATH_RECENT"]
os.makedirs(scaled_recent, exist_ok=True)

# Laden des auf historischen Daten trainierten Scalers
scaler_path = "../../../scaler/scaler.joblib"
scaler = joblib.load(scaler_path)

# Laden des aktuellen dataComplete-Datensatzes
df = pd.read_parquet(data_complete_path).copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Bestimmen der Feature-Liste aus dem Scaler
if not hasattr(scaler, "feature_names_in_"):
    raise AttributeError(
          "Der Scaler enthält keine Feature-Namen. "
        "In diesem Fall muss die Feature-Liste manuell geprüft werden."
    )
feature_cols = list(scaler.feature_names_in_)

# Sicherstellen, dass alle benötigten Features vorhanden sind
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Im dataComplete fehlen folgende benötigte Features: {missing}"
    )

# Erzeugen der für das Backtesting benötigten Dateien
base_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]

x_test = df[base_cols].copy()

# y_test: Zielvariable
y_test = df[["trend"]].copy()

# index_map: Zuordnung von Zeilenindex zu Zeitstempel
x_test_index_map = pd.DataFrame({
    "row_id": range(len(df)),
    "timestamp": df["timestamp"].values
})

# Skalierung der Features (nur transform, kein Fit)
X = df[feature_cols].copy()
X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

# Einheitliche Indizes für alle Ausgabedaten
X_scaled.index = range(len(df))
x_test.index = X_scaled.index
y_test.index = X_scaled.index

# Speichern der Dateien für das Backtesting
X_scaled.to_parquet(f"{scaled_recent}/x_test_scaled.parquet")
x_test.to_parquet(f"{scaled_recent}/x_test.parquet")
y_test.to_parquet(f"{scaled_recent}/y_test.parquet")
x_test_index_map.to_parquet(f"{scaled_recent}/x_test_index_map.parquet", index=False)





