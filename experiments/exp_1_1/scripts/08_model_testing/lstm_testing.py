import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score
from experiments.exp_1_1.scripts.model_training.BTCSequenceDataset import BTCSequenceDataset

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer_1 = nn.LSTM(input_size=input_size, hidden_size=16, batch_first=True)
        self.layer_2 = nn.Linear(16, 3)

    #x hat die Form (batch_size, seq_len, input_size)
    def forward(self, x):
        out, _ = self.layer_1(x)
        #letzter Hidden State als Zusammenfassung der gesamten Sequenz
        out = out[:, -1, :]
        #Input Hidden State
        #Output: Für jede Sequenz 3 Zahlen —> Score für jede mögliche Klasse, 3-dimensionales Output pro Sequenz
        out = self.layer_2(out)
        return out

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
input_size = params['MODELING']['INPUT_SIZE']
model_path = params['MODELING']['SAVE_MODEL']
scaled_path = params['POST_SPLIT']['SCALED_PATH']

x_test = pd.read_parquet(f'{scaled_path}/x_test_scaled.parquet')
y_test = pd.read_parquet(f'{scaled_path}/y_test.parquet')

seq = params['MODELING']['SEQUENCE']
batch_size = params['MODELING']['BATCH_SIZE']

test_dataset = BTCSequenceDataset(
    x=x_test,
    y=y_test,
    seq_size=seq
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

net_test = Net(input_size)
criterion = nn.CrossEntropyLoss()
#Modellgewichte laden
net_test.load_state_dict(torch.load(f"{model_path}/best_model.pt"))


net_test.eval()
runTest_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, labels, _ in test_loader:
        outputs = net_test(inputs)
        loss = criterion(outputs, labels)
        runTest_loss += loss.item()
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

test_loss = runTest_loss / len(test_loader)
all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

test_acc = accuracy_score(all_targets, all_preds)
test_f1 = f1_score(all_targets, all_preds, average="macro")
test_recall = recall_score(all_targets, all_preds, average="macro")

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1: {test_f1:.4f}")


# Konfusionsmatrix berechnen
cm = confusion_matrix(all_targets, all_preds)

labels = ["DOWN", "NEUTRAL", "UP"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – LSTM")
plt.tight_layout()
plt.show()

