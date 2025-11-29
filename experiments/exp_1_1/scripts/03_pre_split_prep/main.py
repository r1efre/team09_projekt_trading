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

#Load data
rawDataPath = f"{data_path}/{raw_data_file}"
raw_data = pd.read_parquet(rawDataPath)

# Prüfen auf NaN Werte
print(raw_data.isna().sum())

#Nach Zeit sortieren
raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
raw_data = raw_data.sort_values("timestamp").reset_index(drop=True)

# Features generieren
data_with_features = features.generate_features(raw_data)

# Targets generieren
data_complete = targets.set_target(data_with_features)


