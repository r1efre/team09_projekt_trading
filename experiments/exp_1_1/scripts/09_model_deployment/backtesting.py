import yaml
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from experiments.exp_1_1.scripts.model_training.BTCSequenceDataset import BTCSequenceDataset

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer_1 = nn.LSTM(input_size=input_size, hidden_size=16, batch_first=True)
        self.layer_2 = nn.Linear(16, 3)

    def forward(self, x):
        out, _ = self.layer_1(x)
        out = out[:, -1, :]
        out = self.layer_2(out)
        return out

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
input_size = params['MODELING']['INPUT_SIZE']
model_path = params['MODELING']['SAVE_MODEL']
scaled_path = params['POST_SPLIT']['SCALED_PATH']

x_test_scaled = pd.read_parquet(f'{scaled_path}/x_test_scaled.parquet')
x_test = pd.read_parquet(f'{scaled_path}/x_test.parquet')
y_test = pd.read_parquet(f'{scaled_path}/y_test.parquet')
mapping = pd.read_parquet(f'{scaled_path}/x_test_index_map.parquet')
mapping["timestamp"] = pd.to_datetime(mapping["timestamp"])
mapping = mapping.set_index("row_id")

seq = params['MODELING']['SEQUENCE']
batch_size = params['MODELING']['BATCH_SIZE']

test_dataset = BTCSequenceDataset(
    x=x_test_scaled,
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

price_series = x_test["close"]
price_series.index = mapping.loc[price_series.index, "timestamp"].values
price_series.index = pd.to_datetime(price_series.index)

account_model = 100000
positions_model = []
equity_curve = []
equity_timestamps = []
trades_count = 0
trades = {}
net_test.eval()

def calculate_equity(row_index, positions, account):
    current_price = x_test.loc[row_index, 'close']
    if len(positions) > 0:
        position_value = positions[0]['shares'] * current_price
    else:
        position_value = 0.0

    equity = account + position_value
    return equity


def setOrder(predicted, row_index, account, positions, timestamp):
    current_price = x_test.loc[row_index, 'close']

    if predicted == 2:  # UP - KAUFEN
        trades[timestamp] = "BUY"
        if len(positions) == 0:
            # Erste Position öffnen
            position_size = account * 0.1
            shares = position_size / current_price
            account = account - position_size

            positions.append({
                'entry_price': current_price,
                'entry_index': row_index,
                'shares': shares,
                'position_size': position_size,
                'buys': 1  # Anzahl Käufe tracken
            })

            print(f"🟢 BUY (Initial) at {row_index}")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Shares: {shares:.4f}")
            print(f"   Account: {account:.4f}")

        else:
            # Position bereits offen - NACHKAUFEN
            position = positions[0]

            additional_size = account * 0.05  # Nur 5% nachkaufen
            additional_shares = additional_size / current_price
            account = account - additional_size

            # Position updaten
            total_shares = position['shares'] + additional_shares
            total_investment = position['position_size'] + additional_size

            position['shares'] = total_shares
            position['position_size'] = total_investment
            position['buys'] += 1

            print(f"🟢 BUY (Add-on #{position['buys']}) at {row_index}")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Additional Shares: {additional_shares:.4f}")
            print(f"   Total Shares: {total_shares:.4f}")

    elif predicted == 0:  # DOWN - VERKAUFEN
        trades[timestamp] = "SELL"
        if len(positions) > 0:  # Nur verkaufen wenn Position offen
            position = positions[0]
            entry_price = position['entry_price']
            shares = position['shares']

            # Gewinn/Verlust berechnen
            exit_value = shares * current_price
            entry_value = position['position_size']
            profit = exit_value - entry_value
            profit_pct = (profit / entry_value) * 100

            # Kapital updaten
            account += exit_value

            print(f"🔴 SELL at {row_index}")
            print(f"   Entry: ${entry_price:.2f} → Exit: ${current_price:.2f}")
            print(f"   Shares: {shares:.4f}")
            print(f"   Profit: ${profit:.2f} ({profit_pct:+.2f}%)")
            print(f"   New Account: ${account:.2f}")

            # Position schließen
            positions.clear()

    if len(positions) > 0:
        position_value_after = positions[0]['shares'] * current_price
    else:
        position_value_after = 0.0
    equity_after = account + position_value_after
    return account, equity_after


with torch.no_grad():
    for inputs, labels, indices in test_loader:
        outputs = net_test(inputs)
        probs = torch.softmax(outputs, dim=1)

        for i in range(probs.shape[0]):
            row_index = indices[i].item()

            prob_down = probs[i, 0].item()
            prob_hold = probs[i, 1].item()
            prob_up = probs[i, 2].item()

            predicted = torch.argmax(probs[i]).item()

            # Nur wenn Up oder Down predicted (nicht Hold)
            if predicted in [0, 2]:  # 0=Down, 2=Up
                diff = abs(prob_up - prob_down) * 100

                if diff >= 5:
                    print("---Model---")
                    timestamp = mapping.loc[row_index, "timestamp"]
                    equity_timestamps.append(timestamp)
                    account_model, equity = setOrder(predicted, row_index, account_model, positions_model, timestamp)
                    equity_curve.append(equity)
                    trades_count += 1

                else:
                    equity = calculate_equity(row_index, positions_model, account_model)
                    equity_curve.append(equity)
                    timestamp = mapping.loc[row_index, "timestamp"]
                    equity_timestamps.append(timestamp)

            else:
                equity = calculate_equity(row_index, positions_model, account_model)
                equity_curve.append(equity)
                timestamp = mapping.loc[row_index, "timestamp"]
                equity_timestamps.append(timestamp)


print("-----------------------------------------------")
print(f"   Account Model: {account_model:.4f}")


equity_series = pd.Series(
    equity_curve,
    index=pd.to_datetime(equity_timestamps)
).sort_index()

plt.figure(figsize=(12, 5))
plt.plot(equity_series, label="Equity Curve")

plt.title("Equity Curve – Backtesting")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()









