import yaml
import pandas as pd

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = params['DATA_PREP']['PROCESSED_PATH']
data = pd.read_parquet(f'{processed_path}/dataComplete.parquet')

# Unpack date boundaries for train/validation/test splits.
train_date = params['SPLIT_DATA']['TRAIN_DATE']
validation_date = params['SPLIT_DATA']['VALIDATION_DATE']

# Split into train, validation, and test sets and save the processed data to Parquet files
train = data[data['timestamp'] < train_date]
train.to_parquet(f"{processed_path}/train.parquet", index=False)

validation = data[(data['timestamp'] >= train_date) & (data['timestamp'] < validation_date)]
validation.to_parquet(f"{processed_path}/validation.parquet", index=False)

test = data[(data['timestamp'] >= validation_date)]
test.to_parquet(f"{processed_path}/test.parquet", index=False)
