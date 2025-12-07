import yaml
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from .BTCSequenceDataset import BTCSequenceDataset


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
scaled_path = params['POST_SPLIT']['SCALED_PATH']
model_path = params['MODELING']['SAVE_MODEL']

x_train = pd.read_parquet(f'{scaled_path}/x_train_scaled.parquet')
y_train = pd.read_parquet(f'{scaled_path}/y_train.parquet')
x_val = pd.read_parquet(f'{scaled_path}/x_val_scaled.parquet')
y_val = pd.read_parquet(f'{scaled_path}/y_val.parquet')

seq = params['MODELING']['SEQUENCE']
input_size = params['MODELING']['INPUT_SIZE']
batch_size = params['MODELING']['BATCH_SIZE']

#Definition von Sequenzen
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
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
)

net = Net(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainLoss_vals = []
ValLoss_vals = []
val_acc_vals = []
val_f1_vals = []
val_recall_vals = []

num_epochs = 70
best_val_loss = 2
for epoch in range(num_epochs):
    # Training
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
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
        for inputs, labels in val_loader:
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
        torch.save(net.state_dict(), f"{model_path}/best_model.pt")
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
        f"Val Acc: { val_acc_vals[-1]:.3f} | "
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
plt.show()

