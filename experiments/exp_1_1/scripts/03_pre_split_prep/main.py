import os
import pandas as pd
import yaml
import targets, features

params = yaml.safe_load(open("../../conf/params.yaml"))

# Unpack data paths and ensure processed data directory exists.
data_path = params['DATA_ACQUISITION']['DATA_PATH']
processed_path = params['DATA_PREP']['PROCESSED_PATH']
raw_data_file = params['DATA_PREP']['RAW_DATA_FILE']
os.makedirs(processed_path, exist_ok=True)

# Unpack relevant parameters for feature calculation.
ema_periods = params['DATA_PREP']['EMA_PERIODS']
return_periods = params['DATA_PREP']['RETURN_PERIODS']
rsi_atr_window = params['DATA_PREP']['RSI_ATR_WINDOW']

#Load data
rawDataPath = f"{data_path}/{raw_data_file}"
raw_data = pd.read_parquet(rawDataPath)

# Prüfen auf NaN Werte
print(raw_data.isna().sum())

#Nach Zeit sortieren
raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
raw_data = raw_data.sort_values("timestamp").reset_index(drop=True)

# Features generieren
data_with_features, features = features.generate_features(raw_data, return_periods, ema_periods, rsi_atr_window)

# Targets generieren
data_complete = targets.set_target(data_with_features)

#NaN Werte entfernen, die durch das Feature Engineering entstanden sind
data_complete = data_complete.dropna().reset_index(drop=True)
print(data_complete.isna().sum())

data_complete.to_parquet(f'{processed_path}/dataComplete.parquet', index=False)

if not os.path.exists("features.txt"):
    with open("features.txt", "w") as f:
        for feature in features:
            f.write(f"{feature}\n")



