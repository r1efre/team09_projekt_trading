import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import yaml

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
scaled_path = params['POST_SPLIT']['SCALED_PATH']

x_train = pd.read_parquet(f'{scaled_path}/x_train.parquet')
y_train = pd.read_parquet(f'{scaled_path}/y_train.parquet')
x_val = pd.read_parquet(f'{scaled_path}/x_val.parquet')
y_val = pd.read_parquet(f'{scaled_path}/y_val.parquet')

rfc = RandomForestClassifier(
    n_estimators=200, #Bäume
    max_depth=10, #Tiefe
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced', #Klasse proportional zur Häufigkeit gewichtet
    random_state=42,
    n_jobs=-1,
    verbose=1,
    oob_score=True
)

rfc.fit(x_train, y_train)  # Nutzt unskalierte Daten
y_val_pred_rfc = rfc.predict(x_val)

accuracy_rfc = accuracy_score(y_val, y_val_pred_rfc)
f1_macro_rfc = f1_score(y_val, y_val_pred_rfc, average='macro')
recall_macro_rfc = recall_score(y_val, y_val_pred_rfc, average='macro')

print("Accuracy:", accuracy_rfc)
print("F1-macro:", f1_macro_rfc)
print("Recall-macro:", recall_macro_rfc)


