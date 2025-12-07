import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from experiments.exp_1_1.scripts.model_training.LSTM_Pytorch import Net
from experiments.exp_1_1.scripts.model_training.BTCSequenceDataset import BTCSequenceDataset

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
net_test.load_state_dict(torch.load(f"{model_path}/best_model.pt"))


net_test.eval()
runTest_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
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
