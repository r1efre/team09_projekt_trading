import yaml
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from BTCSequenceDataset import BTCSequenceDataset


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        # GRÖSSERE Hidden Sizes möglich!
        self.layer_1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True)
        self.dropout_1 = nn.Dropout(0.3)
        self.layer_2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout_2 = nn.Dropout(0.3)

        # Extra Dense Layers
        self.layer_3 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.3)
        self.layer_4 = nn.Linear(32, 3)

    def forward(self, x):
        out, _ = self.layer_1(x)
        out = self.dropout_1(out)
        out, _ = self.layer_2(out)
        out = out[:, -1, :]
        out = self.dropout_2(out)
        out = self.layer_3(out)
        out = self.relu(out)
        out = self.dropout_3(out)
        out = self.layer_4(out)
        return out


# Daten laden
BASE_DIR = Path(__file__).resolve().parent
params_path = BASE_DIR / "../../conf/params.yaml"
params = yaml.safe_load(open(params_path))
scaled_path = params['POST_SPLIT']['SCALED_PATH']

ROOT_DIR = Path(__file__).resolve().parents[2]
# Modell-Verzeichnis
model_dir = ROOT_DIR / "model"
model_dir.mkdir(parents=True, exist_ok=True)

x_train = pd.read_parquet(f'{scaled_path}/x_train_scaled.parquet')
y_train = pd.read_parquet(f'{scaled_path}/y_train.parquet')
x_val = pd.read_parquet(f'{scaled_path}/x_val_scaled.parquet')
y_val = pd.read_parquet(f'{scaled_path}/y_val.parquet')

seq = params['MODELING']['SEQUENCE']
input_size = params['MODELING']['INPUT_SIZE']
batch_size = params['MODELING']['BATCH_SIZE']

# Definition von Sequenzen
train_dataset = BTCSequenceDataset(
    x=x_train,
    y=y_train,
    seq_size=seq
)

val_dataset = BTCSequenceDataset(
    x=x_val,
    y=y_val,
    seq_size=seq
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size)

net = Net(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainLoss_vals = []
ValLoss_vals = []
val_acc_vals = []
val_f1_vals = []
val_recall_vals = []

num_epochs = 20
best_val_loss = 2
for epoch in range(num_epochs):
    # Training
    net.train()
    running_loss = 0.0
    for inputs, labels, _ in train_loader:
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    trainLoss_vals.append(train_loss)

    # Validation
    net.eval()
    runVal_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            runVal_loss += loss.item()

            # Klassen vorhersagen (Argmax über die 3 Logits)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    val_loss = runVal_loss / len(val_loader)
    ValLoss_vals.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(net.state_dict(), model_dir / "best_model.pt")
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # alles zu einem Vektor zusammenfügen
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Metriken berechnen
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average="macro")
    val_recall = recall_score(all_targets, all_preds, average="macro")

    val_acc_vals.append(val_acc)
    val_f1_vals.append(val_f1)
    val_recall_vals.append(val_recall)

print(
    f"Val Acc: {val_acc_vals[-1]:.3f} | "
    f"Val F1: {val_f1_vals[-1]:.3f} | "
    f"Val Recall: {val_recall_vals[-1]:.3f}"
)

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), trainLoss_vals, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), ValLoss_vals, marker='s', label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("CrossEntropyLoss")
plt.title("Training und Validation Loss über Epochen")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("../../images/val_train_loss.png")
plt.close()

