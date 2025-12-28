import os
from pathlib import Path
import pandas as pd
import sys
import yaml


# Pfad zur Pre-Split-Preparation setzen
PREP_DIR = Path(__file__).resolve().parents[2] / "03_pre_split_prep"
sys.path.insert(0, str(PREP_DIR))
import targets as targ
import features as feat

# Laden der Projektparameter aus der Konfigurationsdatei
params = yaml.safe_load(open("../../../conf/params.yaml"))

# Pfade für die neuen (recent) Roh- und Prozessdaten
data_path = params['BACKTESTING_RECENT']['DATA_PATH_RECENT']
processed_path = params['BACKTESTING_RECENT']['PROCESSED_PATH_RECENT']
raw_data_file = params['DATA_PREP']['RAW_DATA_FILE']
os.makedirs(processed_path, exist_ok=True)

# Parameter für das Feature Engineering
ema_periods = params['DATA_PREP']['EMA_PERIODS']
return_periods = params['DATA_PREP']['RETURN_PERIODS']
rsi_atr_window = params['DATA_PREP']['RSI_ATR_WINDOW']

# Laden der zusammengeführten Rohdaten (BTC + ETH)
rawDataPath = f"{data_path}/{raw_data_file}"
raw_data = pd.read_parquet(rawDataPath)

# Prüfung auf fehlende Werte in den Rohdaten
print(raw_data.isna().sum())

# Zeitstempel in Datumsformat umwandeln und Daten sortieren
raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
raw_data = raw_data.sort_values("timestamp").reset_index(drop=True)

# Generierung der technischen Features
data_with_features, features = feat.generate_features(raw_data, return_periods, ema_periods, rsi_atr_window)

# Erzeugen der Zielvariable (Trend: up / down / neutral)
data_complete = targ.set_target(data_with_features)

# Entfernen von NaN-Werten, die durch Feature Engineering entstehen
data_complete = data_complete.dropna().reset_index(drop=True)
print(data_complete.isna().sum())

# Speichern des vollständig vorbereiteten Datensatzes
# für das Backtesting der aktuellen Daten
data_complete.to_parquet(f'{processed_path}/dataComplete_recent.parquet', index=False)

