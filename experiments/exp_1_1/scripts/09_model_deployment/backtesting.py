import yaml
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from onnxruntime.transformers.models.llama.dist_settings import print_out
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

# Overall performance metrics
final_equity = equity_series.iloc[-1]
initial_capital = 100000

absolute_return = final_equity - initial_capital
relative_return = absolute_return / initial_capital * 100

start_date = equity_series.index.min()
end_date = equity_series.index.max()

print("Final capital:", final_equity)
print("Absolute return:", absolute_return)
print("Relative return (%):", relative_return)
print("Trades:", trades_count)
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



# Trades series (timestamp - BUY/SELL)
# Das trades-Dictionary speichert alle von der Strategie generierten BUY- und SELL-Aktionen.
# Es wird verwendet, um Zeitpunkt, Häufigkeit und zeitliche Verteilung der Handelsentscheidungen zu analysieren.

trades_series = pd.Series(trades)
trades_series.index = pd.to_datetime(trades_series.index)
trades_series = trades_series.sort_index()

# Remove timezone from trades_series
if getattr(trades_series.index, "tz", None) is not None:
    trades_series.index = trades_series.index.tz_localize(None)



# Monthly count of BUY/SELL actions
buy_per_month = (trades_series == "BUY").resample("ME").sum()
sell_per_month = (trades_series == "SELL").resample("ME").sum()

plt.figure(figsize=(12, 4))
plt.plot(buy_per_month.index, buy_per_month.values, label="BUY-Anzahl")
plt.plot(sell_per_month.index, sell_per_month.values, label="SELL-Anzahl")
plt.title("BUY vs SELL-Aktionen im Zeitverlauf (monatlich)")
plt.xlabel("Zeit")
plt.ylabel("Anzahl")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("../../images/09_buy_sell_actions.png")
plt.close()



# Price series (timestamp -> close)
# Die Price-Series repräsentiert die historischen BTC-Schlusskurse mit Zeitstempel.
# Sie dient als Referenz für die Marktentwicklung und zur Visualisierung von Handelszeitpunkten.
price_series = pd.Series(
    data=x_test["close"].values,
    index=pd.to_datetime(mapping.loc[x_test.index, "timestamp"].values)
).sort_index()

# Make price timestamps tz-naive for consistent plotting
if getattr(price_series.index, "tz", None) is not None:
    price_series.index = price_series.index.tz_convert(None)



# Price + BUY/SELL signals
WINDOW = "5D"          # "1D", "3D", "7D"
MIN_TRADES = 3         # minimum number of actions required to plot a window

if trades_series.empty:
    raise ValueError("No trades found: trades_series is empty.")

trade_times = trades_series.index.sort_values()
trade_counts = pd.Series(1, index=trade_times)
rolling_counts = trade_counts.rolling(WINDOW).sum()

best_end = rolling_counts.idxmax()
best_count = int(rolling_counts.loc[best_end])

if best_count < MIN_TRADES:
    center = trade_times[0]
    best_start = center - pd.Timedelta(WINDOW) / 2
    best_end = center + pd.Timedelta(WINDOW) / 2
else:
    best_start = best_end - pd.Timedelta(WINDOW)

price_window = price_series.loc[best_start:best_end]
trades_window = trades_series.loc[best_start:best_end]

buy_times_w = trades_window[trades_window == "BUY"].index
sell_times_w = trades_window[trades_window == "SELL"].index

buy_prices = price_window.reindex(buy_times_w, method="nearest") if len(buy_times_w) else pd.Series(dtype=float)
sell_prices = price_window.reindex(sell_times_w, method="nearest") if len(sell_times_w) else pd.Series(dtype=float)

plt.figure(figsize=(12, 5))
plt.plot(price_window, label="BTC Close", linewidth=1.5)

if not buy_prices.empty:
    plt.scatter(buy_prices.index, buy_prices.values, marker="^", s=90, label="BUY")

if not sell_prices.empty:
    plt.scatter(sell_prices.index, sell_prices.values, marker="v", s=90, label="SELL")

plt.title("Preis mit BUY-/SELL-Signalen")
plt.xlabel("Zeit")
plt.ylabel("Preis")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("../../images/09_price_actions.png")
plt.close()



# Weekly count of BUY/SELL actions
actions_per_week = trades_series.resample("W").size()

plt.figure(figsize=(12, 4))
plt.bar(actions_per_week.index.astype(str), actions_per_week.values)
plt.xlabel("Woche")
plt.ylabel("Anzahl")
plt.title("Histogramm der Handelsaktivitäten pro Woche")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("../../images/09_trades_per_week.png")
plt.close()



# Count of BUY/SELL actions
counts = trades_series.value_counts().reindex(["BUY", "SELL"]).fillna(0)

plt.figure(figsize=(5, 4))
plt.bar(
    counts.index,
    counts.values,
    color=["tab:blue", "tab:orange"]
)
plt.xlabel("Aktion")
plt.ylabel("Anzahl")
plt.title("BUY vs SELL-Anzahl")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("../../images/09_buy_vs_sell.png")
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
plt.savefig("../../images/09_btc_price_equity_comparision.png")
plt.close()


