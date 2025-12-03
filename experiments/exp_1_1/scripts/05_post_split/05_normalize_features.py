import yaml
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# Define the path to the data directory
art = Path("../../scaler")

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = params['DATA_PREP']['PROCESSED_PATH']
scaled_path = params['POST_SPLIT']['SCALED_PATH']

train = pd.read_parquet(f'{processed_path}/train.parquet').set_index("timestamp")
val = pd.read_parquet(f'{processed_path}/validation.parquet').set_index("timestamp")
test = pd.read_parquet(f'{processed_path}/test.parquet').set_index("timestamp")

# Specify the target variable and feature columns
target = "trend"
feature_cols = [c for c in train.columns if c != target]

# Separate features (X) and target (y) for each split
X_train, y_train = train[feature_cols], train[target]
X_val, y_val = val[feature_cols], val[target]
X_test, y_test = test[feature_cols], test[target]

# Initialize a StandardScaler and fit only on the training data to avoid data leakage
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    index=X_train.index,
    columns=feature_cols
)

X_val_scaled = scaler.transform(X_val)
X_val_scaled = pd.DataFrame(
    X_val_scaled,
    index=X_val.index,
    columns=feature_cols
)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(
    X_test_scaled,
    index=X_test.index,
    columns=feature_cols
)

# Save unscaled feature and target splits
X_train.to_parquet(f"{scaled_path}/x_train.parquet", index=False)
pd.DataFrame(y_train).to_parquet(f"{scaled_path}/y_train.parquet", index=False)
X_val.to_parquet(f"{scaled_path}/x_val.parquet", index=False)
pd.DataFrame(y_val).to_parquet(f"{scaled_path}/y_val.parquet", index=False)
X_test.to_parquet(f"{scaled_path}/x_test.parquet", index=False)
pd.DataFrame(y_test).to_parquet(f"{scaled_path}/y_test.parquet", index=False)

# Save scaled feature splits
X_train_scaled.to_parquet(f"{scaled_path}/x_train_scaled.parquet", index=False)
X_val_scaled.to_parquet(f"{scaled_path}/x_val_scaled.parquet", index=False)
X_test_scaled.to_parquet(f"{scaled_path}/x_test_scaled.parquet", index=False)

# Save the fitted scaler object for later use
joblib.dump(scaler, art / "scaler.joblib")