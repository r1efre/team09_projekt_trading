import yaml
import pandas as pd, matplotlib.pyplot as plt


#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
scaled_path = params['POST_SPLIT']['SCALED_PATH']

# Load train data
x_train_scaled = pd.read_parquet(f'{scaled_path}/x_train_scaled.parquet')
x_val_scaled = pd.read_parquet(f'{scaled_path}/x_val_scaled.parquet')
x_test_scaled = pd.read_parquet(f'{scaled_path}/x_test_scaled.parquet')

x_train = pd.read_parquet(f'{scaled_path}/x_train.parquet')
x_val = pd.read_parquet(f'{scaled_path}/x_val.parquet')
x_test = pd.read_parquet(f'{scaled_path}/x_test.parquet')

# Calculate correlation matrix
corr_matrix = x_train_scaled.corr()

# Plot correlation heatmap
plt.figure(figsize=(10,8), dpi=100)
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# open, high, low, vwap, eth_close  löschen
# Liste der zu entfernenden Spalten
columns_to_drop = ['open', 'high', 'low', 'vwap', 'eth_close']

# Original-Daten bereinigen
X_train = x_train.drop(columns=columns_to_drop, errors='ignore')
X_val = x_val.drop(columns=columns_to_drop, errors='ignore')
X_test = x_test.drop(columns=columns_to_drop, errors='ignore')

# Skalierte Daten ebenfalls bereinigen
X_train_scaled = x_train_scaled.drop(columns=columns_to_drop, errors='ignore')
X_val_scaled = x_val_scaled.drop(columns=columns_to_drop, errors='ignore')
X_test_scaled = x_test_scaled.drop(columns=columns_to_drop, errors='ignore')

# Save scaled feature splits
X_train_scaled.to_parquet(f"{scaled_path}/x_train_scaled.parquet", index=False)
X_val_scaled.to_parquet(f"{scaled_path}/x_val_scaled.parquet", index=False)
X_test_scaled.to_parquet(f"{scaled_path}/x_test_scaled.parquet", index=False)

# Save unscaled feature
X_train.to_parquet(f"{scaled_path}/x_train.parquet", index=False)
X_val.to_parquet(f"{scaled_path}/x_val.parquet", index=False)
X_test.to_parquet(f"{scaled_path}/x_test.parquet", index=False)

