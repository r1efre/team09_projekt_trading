import yaml
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from experiments.exp_1_1.scripts.model_training.BTCSequenceDataset import BTCSequenceDataset

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
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

#Daten laden
params = yaml.safe_load(open("../../conf/params.yaml"))
input_size = params['MODELING']['INPUT_SIZE']
scaled_path = params['POST_SPLIT']['SCALED_PATH']

ROOT_DIR = Path(__file__).resolve().parents[2]
model_dir = ROOT_DIR / "model"
model_dir.mkdir(parents=True, exist_ok=True)

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
net_test.load_state_dict(torch.load(f"{model_dir}/best_model.pt"))

price_series = x_test["close"]
price_series.index = mapping.loc[price_series.index, "timestamp"].values
price_series.index = pd.to_datetime(price_series.index)

account_model = 100000
positions_model = []
equity_curve = []
equity_timestamps = []
signals_count = 0
buying_count = 0
trades = {}
net_test.eval()
holding_count = 0

def calculate_equity(row_index, positions, account):
    current_price = x_test.loc[row_index, 'close']
    if len(positions) > 0:
        position_value = positions[0]['shares'] * current_price
    else:
        position_value = 0.0

    equity = account + position_value
    return equity

# Erfassung der Transaktionsrendite
trade_log = []
# Erfassung der Aktionen BUY/SELL
action_log = []

def setSellOrder(row_index, account, positions, timestamp, buying_count):
    current_price = x_test.loc[row_index, 'close']
    trades[timestamp] = "SELL"
    if len(positions) > 0:  # Nur verkaufen wenn Position offen
        position = positions[0]
        entry_price = position['entry_price']
        shares = position['shares']

        action_log.append({
                "timestamp": timestamp,
                "action": "SELL"
        })
        buying_count = 0
        # Gewinn/Verlust berechnen
        exit_value = shares * current_price
        position_size = position['position_size']
        profit = exit_value - position_size
        profit_pct = (profit / position_size) * 100

        trade_log.append({
            "timestamp": timestamp,
            "type": "SELL",
            "entry_index": position["entry_index"],
            "exit_index": row_index,
            "entry_price": entry_price,
            "exit_price": current_price,
            "position_size": position_size,
            "exit_value": exit_value,
            "profit": profit,
            "profit_pct": profit_pct
        })

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
    return account, equity_after, buying_count

def setBuyOrder(prob_up, row_index, account, positions, timestamp, buying_count):
    current_price = x_test.loc[row_index, 'close']

    if prob_up >= 0.5:
        buy_pct = 0.2
    elif prob_up >= 0.4:
        buy_pct = 0.15
    elif prob_up >= 0.3:
        buy_pct = 0.1
    else:
        buy_pct = 0.05

    if buying_count < 11:  # UP - KAUFEN
        trades[timestamp] = "BUY"
        if len(positions) == 0:
            # Erste Position öffnen
            position_size = account * buy_pct
            shares = position_size / current_price #wie viel Prozent vom Bitcoin Preis investiert man
            account = account - position_size

            positions.append({
                'entry_price': current_price,
                'entry_index': row_index,
                'shares': shares,
                'position_size': position_size,
                'buys': 1  # Anzahl Käufe tracken
            })
            buying_count += 1
            action_log.append({
                "timestamp": timestamp,
                "action": "BUY"
            })

            print(f"🟢 BUY (Initial) at {row_index}")
            print(f"   Timestamp: ${timestamp}")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Position: ${position_size:.2f}")
            print(f"   Shares: {shares:.4f}")
            print(f"   Account: {account:.4f}")

        else:
            # Position bereits offen - NACHKAUFEN
            position = positions[0]

            additional_size = account * buy_pct  # Nur 5% nachkaufen
            additional_shares = additional_size / current_price
            account = account - additional_size

            # Position updaten
            total_shares = position['shares'] + additional_shares
            total_investment = position['position_size'] + additional_size

            position['shares'] = total_shares
            position['position_size'] = total_investment
            position['buys'] += 1
            buying_count += 1
            action_log.append({
                "timestamp": timestamp,
                "action": "BUY_ADD"
            })

            print(f"🟢 BUY (Add-on #{position['buys']}) at {row_index}")
            print(f"   Timestamp: ${timestamp}")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Position: ${additional_size:.2f}")
            print(f"   Additional Shares: {additional_shares:.4f}")
            print(f"   Total Shares: {total_shares:.4f}")

    else:
        print("BUYING SIGNAL -- BUT BOUGHT ALREADY 15 TIMES")

    if len(positions) > 0:
        position_value_after = positions[0]['shares'] * current_price
    else:
        position_value_after = 0.0
    equity_after = account + position_value_after
    return account, equity_after, buying_count

predictions_up = []
predictions_hold = []
predictions_down = []
predictions = 0
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
            predictions += 1
            predictions_up.append(prob_up)
            predictions_hold.append(prob_hold)
            predictions_down.append(prob_down)

            timestamp = mapping.loc[row_index, "timestamp"]
            ts = pd.Timestamp(timestamp)
            if predictions == 15:

                avg_prob_down = np.mean(predictions_down)
                avg_prob_up = np.mean(predictions_up)
                avg_prob_hold = np.mean(predictions_hold)

                avg_probs = {
                    0: avg_prob_down,
                    1: avg_prob_hold,
                    2: avg_prob_up
                }

                max_class = max(avg_probs, key=avg_probs.get)

                predictions_up.clear()
                predictions_hold.clear()
                predictions_down.clear()
                predictions = 0
                # Nur wenn Up oder Down predicted (nicht Hold)
                if (max_class == 2) or (holding_count > 10 and avg_prob_down < avg_prob_up):  # 0=Down, 2=Up
                    print("---Model---")
                    equity_timestamps.append(timestamp)
                    account_model, equity, buying_count = setBuyOrder(avg_prob_up, row_index, account_model, positions_model, timestamp, buying_count)
                    equity_curve.append(equity)
                    signals_count += 1
                    holding_count = 0

                elif (max_class == 0 ) or (holding_count > 10 and avg_prob_down > avg_prob_up):
                    print("---Model---")
                    equity_timestamps.append(timestamp)
                    account_model, equity, buying_count = setSellOrder(row_index, account_model, positions_model,timestamp, buying_count)
                    equity_curve.append(equity)
                    holding_count = 0

                else:
                    print("---HOLD---")
                    holding_count += 1
                    equity = calculate_equity(row_index, positions_model, account_model)
                    equity_curve.append(equity)
                    equity_timestamps.append(timestamp)
            else:
                equity = calculate_equity(row_index, positions_model, account_model)
                equity_curve.append(equity)
                equity_timestamps.append(timestamp)

print("-----------------------------------------------")
print(f"   Account Model: {account_model:.4f}")

equity_series = pd.Series(
    equity_curve,
    index=pd.to_datetime(equity_timestamps)
).sort_index()

# Durchschnittliche Anzahl der Aktionen (BUY + SELL)
# und durchschnittliche Rendite der abgeschlossenen Transaktionen
# für x Stunden
def trades_stats_by_hours(action_log, trade_log, hours=24):

    if len(action_log) == 0:
        return None, {
            "hours": hours,
            "avg_actions_per_bin": 0.0,
            "avg_trade_return_pct_per_bin": None,
            "total_actions": 0
        }

    # ---------- Aktionen (BUY + SELL) ----------
    df_actions = pd.DataFrame(action_log)
    df_actions["timestamp"] = pd.to_datetime(df_actions["timestamp"])
    df_actions = df_actions.sort_values("timestamp").set_index("timestamp")

    # BUY_ADD als BUY zählen
    df_actions["action"] = df_actions["action"].replace({"BUY_ADD": "BUY"})

    rule = f"{int(hours)}h"

    actions_per_bin = (
        df_actions
        .resample(rule)
        .size()
        .rename("actions_count")
    )

    # ---------- Trades (SELL → Rendite) ----------
    if len(trade_log) > 0:
        df_trades = pd.DataFrame(trade_log)
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
        df_trades = df_trades.sort_values("timestamp").set_index("timestamp")

        avg_return_per_bin = (
            df_trades["profit_pct"]
            .resample(rule)
            .mean()
            .rename("avg_profit_pct")
        )
    else:
        avg_return_per_bin = pd.Series(name="avg_profit_pct", dtype=float)

    # ---------- Zusammenführen ----------
    df_bins = pd.concat([actions_per_bin, avg_return_per_bin], axis=1)

    summary = {
        "hours": hours,
        "avg_actions_per_bin": df_bins["actions_count"].mean(),
        "avg_trade_return_pct_per_bin": df_bins["avg_profit_pct"].mean(),
        "total_actions": int(df_bins["actions_count"].sum())
    }

    return df_bins, summary



# Anzahl BUY/SELL nach Fenstern X Stunden
def buy_sell_activity_by_hours(action_log, hours=12, count_buy_add_as_buy=True):

    if len(action_log) == 0:
        return None, {
            "window_hours": hours,
            "avg_buys_per_window": 0.0,
            "avg_sells_per_window": 0.0,
            "total_buys": 0,
            "total_sells": 0
        }

    df = pd.DataFrame(action_log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    if count_buy_add_as_buy:
        df["action"] = df["action"].replace({"BUY_ADD": "BUY"})

    rule = f"{int(hours)}h"

    buy_count = (df["action"] == "BUY").resample(rule).sum().astype(int).rename("buy_count")
    sell_count = (df["action"] == "SELL").resample(rule).sum().astype(int).rename("sell_count")

    df_bins = pd.concat([buy_count, sell_count], axis=1)

    summary = {
        "window_hours": hours,
        "avg_buys_per_window": df_bins["buy_count"].mean(),
        "avg_sells_per_window": df_bins["sell_count"].mean(),
        "total_buys": int(df_bins["buy_count"].sum()),
        "total_sells": int(df_bins["sell_count"].sum()),
    }

    return df_bins, summary


X_HOURS = 258  #  6, 12, 24, 48, ...

df_bins, summary = trades_stats_by_hours(action_log, trade_log, hours=X_HOURS)

print("\n--- Action stats by window ---")
print("Window (hours):", summary["hours"])
print("Avg actions per window:", summary["avg_actions_per_bin"])
print("Avg trade return per window (%):", summary["avg_trade_return_pct_per_bin"])
print("Total actions:", summary["total_actions"])



df_act_bins, act_summary = buy_sell_activity_by_hours(action_log, hours=X_HOURS)

print("\n--- Buy/Sell activity by window ---")
print("Window (hours):", act_summary["window_hours"])
print("Avg buys per window:", act_summary["avg_buys_per_window"])
print("Avg sells per window:", act_summary["avg_sells_per_window"])



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
print("Trades (signals count):", signals_count)
print(f"Period: {start_date} - {end_date}")


# Equity curve
plt.figure(figsize=(12, 5))
plt.plot(equity_series, label="Equity Curve")

plt.title("Equity Curve – Backtesting")
plt.xlabel("Zeit")
plt.ylabel("Equity")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("../../images/09_equity_curve.png")
plt.close()

# BTC price and strategy equity curve
if getattr(equity_series.index, "tz", None) is not None:
    equity_series.index = equity_series.index.tz_convert(None)

common_index = price_series.index.intersection(equity_series.index)
price_aligned = price_series.loc[common_index]
equity_aligned = equity_series.loc[common_index]

plt.figure(figsize=(12, 5))
plt.plot(price_aligned.index, price_aligned.values, label="BTC Close", color="tab:blue", linewidth=1.5)
plt.plot(equity_aligned.index, equity_aligned.values, label="Equity", color="tab:orange", linewidth=2.0)
plt.title("Vergleich von BTC-Preis und Equity-Kurve ")
plt.xlabel("Zeit")
plt.ylabel("Wert")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


