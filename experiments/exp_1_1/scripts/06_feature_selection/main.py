import yaml
import pandas as pd, matplotlib.pyplot as plt


#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
scaled_path = params['POST_SPLIT']['SCALED_PATH']

# Load train data
x_train = pd.read_parquet(f'{scaled_path}/x_train_scaled.parquet')
y_train = pd.read_parquet(f'{scaled_path}/y_train.parquet')

# Calculate correlation matrix
corr_matrix = x_train.corr()

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
