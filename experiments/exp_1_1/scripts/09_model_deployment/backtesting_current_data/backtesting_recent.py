import os
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


# Laden der Konfigurationsparameter und Pfade
params = yaml.safe_load(open("../../../conf/params.yaml", "r", encoding="utf-8"))

input_size = params["MODELING"]["INPUT_SIZE"]
model_path = "../../../model/best_model.pt"
seq = params["MODELING"]["SEQUENCE"]
batch_size = params["MODELING"]["BATCH_SIZE"]

scaled_recent = params["BACKTESTING_RECENT"]["SCALED_PATH_RECENT"]

# Laden des aktuellen Test-Datensatzes
x_test_scaled = pd.read_parquet(f"{scaled_recent}/x_test_scaled.parquet")
x_test = pd.read_parquet(f"{scaled_recent}/x_test.parquet")
y_test = pd.read_parquet(f"{scaled_recent}/y_test.parquet")
mapping = pd.read_parquet(f"{scaled_recent}/x_test_index_map.parquet")

# timestamp -> datetime
if "timestamp" in mapping.columns:
    mapping["timestamp"] = pd.to_datetime(mapping["timestamp"])

# row_id
if "row_id" in mapping.columns:
    mapping = mapping.set_index("row_id")
else:
    mapping.index.name = "row_id"

# Dataset und DataLoader für Sequenzen
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

# Laden des trainierten Modells
net_test = Net(input_size)
net_test.load_state_dict(torch.load(model_path, map_location="cpu"))
net_test.eval()

# Initialisierung der Trading-Simulation
initial_capital = 100000
account_model = float(initial_capital)
positions_model = []
equity_curve = []
equity_timestamps = []
trades_count = 0

trades = {}
action_log = []
trade_log = []

# Berechnung des aktuellen Portfoliowerts (Cash + offene Position)
def calculate_equity(row_index, positions, account):
    current_price = float(x_test.loc[row_index, "close"])
    position_value = positions[0]["shares"] * current_price if len(positions) > 0 else 0.0
    return account + position_value

# Schließen einer offenen Position (SELL)
def close_position(row_index, account, positions, timestamp):
    # Falls keine Position offen ist
    if len(positions) == 0:
        return account, calculate_equity(row_index, positions, account)

    current_price = float(x_test.loc[row_index, "close"])
    position = positions[0]

    entry_price = float(position["entry_price"])
    shares = float(position["shares"])
    position_size = float(position["position_size"])

    # Gewinn-/Verlustberechnung
    position_value = shares * current_price
    profit_usd = position_value - position_size
    trade_return_pct = (current_price - entry_price) / entry_price * 100.0


    trade_log.append({"timestamp": timestamp, "return_pct": trade_return_pct})

    # Position schließen
    account = account + position_value
    positions.clear()

    action_log.append({"timestamp": timestamp, "action": "SELL"})

    # Konsolenausgabe
    print(f"🔴 SELL at {row_index}")
    print(f"   Entry: ${entry_price:.2f} → Exit: ${current_price:.2f}")
    print(f"   Shares: {shares:.4f}")
    print(f"   Profit: ${profit_usd:.2f} ({trade_return_pct:+.2f}%)")
    print(f"   New Account: ${account:.2f}")

    return account, account


# Handelslogik basierend auf Modellvorhersage
def setOrder(predicted, row_index, account, positions, timestamp):
    current_price = float(x_test.loc[row_index, "close"])

    if predicted == 2:  # UP -> BUY
        trades[timestamp] = "BUY"

        if len(positions) == 0:
            # initial buy
            position_size = account * 0.1
            shares = position_size / current_price
            account -= position_size

            positions.append({
                "entry_price": current_price,
                "entry_index": row_index,
                "shares": shares,
                "position_size": position_size,
                "buys": 1
            })

            action_log.append({"timestamp": timestamp, "action": "BUY"})

            print(f"🟢 BUY (Initial) at {row_index}")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Shares: {shares:.4f}")
            print(f"   Account: {account:.4f}")

        else:
            position = positions[0]
            additional_size = account * 0.05
            additional_shares = additional_size / current_price
            account -= additional_size

            position["shares"] += additional_shares
            position["position_size"] += additional_size
            position["buys"] += 1

            action_log.append({"timestamp": timestamp, "action": "BUY_ADD"})

            print(f"🟢 BUY (Add-on #{position['buys']}) at {row_index}")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Additional Shares: {additional_shares:.4f}")
            print(f"   Total Shares: {position['shares']:.4f}")
            print(f"   Account: {account:.4f}")

        equity_after = calculate_equity(row_index, positions, account)
        return account, equity_after

    if predicted == 0:  # DOWN -> SELL (close if open)
        trades[timestamp] = "SELL"
        account, equity_after = close_position(row_index, account, positions, timestamp)
        return account, equity_after

    # predicted == 1 -> HOLD
    equity_after = calculate_equity(row_index, positions, account)
    return account, equity_after

# Inferenz und Backtesting-Schleife
with torch.no_grad():
    for inputs, labels, indices in test_loader:
        outputs = net_test(inputs)
        probs = torch.softmax(outputs, dim=1)

        for i in range(probs.shape[0]):
            # Jeder Zeitschritt entspricht einer Modellentscheidung
            print("---Model---")

            row_index = int(indices[i].item())

            prob_down = float(probs[i, 0].item())
            prob_up = float(probs[i, 2].item())
            predicted = int(torch.argmax(probs[i]).item())

            timestamp = mapping.loc[row_index, "timestamp"]

            # Handel nur bei Up/Down und ausreichender Konfidenz
            if predicted in [0, 2]:
                diff = abs(prob_up - prob_down) * 100.0
                if diff >= 5.0:
                    account_model, equity = setOrder(
                        predicted, row_index, account_model, positions_model, timestamp
                    )
                    trades_count += 1
                else:
                    equity = calculate_equity(row_index, positions_model, account_model)
            else:
                equity = calculate_equity(row_index, positions_model, account_model)

            equity_curve.append(float(equity))
            equity_timestamps.append(timestamp)

# Ergebnis des Backtestings
equity_series = pd.Series(equity_curve, index=pd.to_datetime(equity_timestamps)).sort_index()
final_equity = float(equity_series.iloc[-1])


# Overall performance metrics
final_equity = equity_series.iloc[-1]
initial_capital = 100000

absolute_return = final_equity - initial_capital
relative_return = absolute_return / initial_capital * 100

start_date = equity_series.index.min()
end_date = equity_series.index.max()

print("\n--- Overall performance metrics ---")
print("Final capital:", final_equity)
print("Absolute return:", absolute_return)
print("Relative return (%):", relative_return)
print("Trades (signals count):", trades_count)
print(f"Period: {start_date} - {end_date}")

# Actions count (BUY/SELL)
actions_df = pd.DataFrame(action_log)

if actions_df.empty:
    buy_count = 0
    sell_count = 0
    buy_add_count = 0
else:
    buy_count = int((actions_df["action"] == "BUY").sum())
    buy_add_count = int((actions_df["action"] == "BUY_ADD").sum())
    sell_count = int((actions_df["action"] == "SELL").sum())

print("\n--- Actions count ---")
print("BUY:", buy_count + buy_add_count)
print("SELL:", sell_count)



# Prepare action log
actions_df = pd.DataFrame(action_log)
actions_df["timestamp"] = pd.to_datetime(actions_df["timestamp"])

# BUY- und SELL-Aktionen trennen
buy_actions = actions_df[actions_df["action"].isin(["BUY", "BUY_ADD"])]
sell_actions = actions_df[actions_df["action"] == "SELL"]

# Zuordnung des Portfolio-Werts zum Zeitpunkt der Aktionen
buy_equity = equity_series.reindex(buy_actions["timestamp"], method="nearest")
sell_equity = equity_series.reindex(sell_actions["timestamp"], method="nearest")

# Plot
plt.figure(figsize=(14, 6))

# Equity-Kurve
plt.plot(equity_series.index, equity_series.values, label="Equity (Portfolio value)", linewidth=2)

# Anfangskapital
plt.axhline(
    initial_capital,
    linestyle="--",
    linewidth=1,
    label="Initial capital"
)

# BUY-Markierungen
plt.scatter(
    buy_equity.index,
    buy_equity.values,
    marker="^",
    s=80,
    label="BUY",
    zorder=5
)

# SELL-Markierungen
plt.scatter(
    sell_equity.index,
    sell_equity.values,
    marker="v",
    s=80,
    label="SELL",
    zorder=5
)

plt.title("Backtest Equity Kurve mit BUY- / SELL-Aktionen")
plt.xlabel("Zeit")
plt.ylabel("Kapital (USD)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("../../../images/09_equity_curve_recent.png")
plt.close()
