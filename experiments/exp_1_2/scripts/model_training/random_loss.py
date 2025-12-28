import yaml
import pandas as pd
import numpy as np

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = params['DATA_PREP']['PROCESSED_PATH']
data = pd.read_parquet(f'{processed_path}/validation.parquet')

#Trend-Verteilung alle 2 Monate

# Trend text labels
label_map = {0: "DOWN", 1: "UP"}
data["trend_label"] = data["trend"].map(label_map)

# Prozentanteil jeder Trendklasse am gesamten Datensatz

# absolute Häufigkeiten
class_counts = data["trend_label"].value_counts()

# prozentuale Verteilung
class_percent = class_counts / len(data)
class_array = class_percent.to_numpy()
print(class_array)

loss_random = -np.sum(class_array * np.log(class_array))
print(loss_random)