import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
scaled_path = params['POST_SPLIT']['SCALED_PATH']
model_path = params['MODELING']['SAVE_MODEL']

x_test = pd.read_parquet(f'{scaled_path}/x_test.parquet')
y_test = pd.read_parquet(f'{scaled_path}/y_test.parquet').squeeze("columns")

rfc = joblib.load(f"{model_path}/randomForest.joblib")

test_pred = rfc.predict(x_test)

accuracy = accuracy_score(y_test.values, test_pred)
f1_macro = f1_score(y_test.values, test_pred, average='macro')
recall_macro = recall_score(y_test.values, test_pred, average='macro')

print("Accuracy:", accuracy)
print("F1-macro:", f1_macro)
print("Recall-macro:", recall_macro)

# Konfusionsmatrix berechnen
cm = confusion_matrix(y_test.values, test_pred)

# Labels für die Achsen
labels = ["DOWN", "NEUTRAL", "UP"]

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Random Forest")
plt.tight_layout()
plt.show()
