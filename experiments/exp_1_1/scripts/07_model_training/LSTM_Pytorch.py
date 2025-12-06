import yaml
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from BTCSequenceDataset import BTCSequenceDataset


class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer_1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.layer_2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.layer_3 = nn.Linear(32, 3)

    #x hat die Form (batch_size, seq_len, input_size)
    def forward(self, x):
        out, _ = self.layer_1(x)
        out, _ = self.layer_2(out)
        out = out[:, -1, :]
        out = self.layer_3(out)
        return out


#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
scaled_path = params['POST_SPLIT']['SCALED_PATH']

x_train = pd.read_parquet(f'{scaled_path}/x_train_scaled.parquet')
y_train = pd.read_parquet(f'{scaled_path}/y_train.parquet')
x_val = pd.read_parquet(f'{scaled_path}/x_val_scaled.parquet')
y_val = pd.read_parquet(f'{scaled_path}/y_val.parquet')

seq = params['MODELING']['SEQUENCE']
input_size = params['MODELING']['INPUT_SIZE']
batch_size = params['MODELING']['BATCH_SIZE']

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
optimizer = optim.Adam(net.parameters(), lr=0.001)

trainLoss_vals = list()
ValLoss_vals = list()
num_epochs = 50
for epoch in range(num_epochs):
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

    # --- Validation ---
    net.eval()
    runVal_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            runVal_loss += loss.item()
    val_loss = runVal_loss / len(val_loader)
    ValLoss_vals.append(val_loss)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), trainLoss_vals, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), ValLoss_vals, marker='s', label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training und Validation Loss über Epochen")
plt.legend()
plt.grid(True)
plt.show()

