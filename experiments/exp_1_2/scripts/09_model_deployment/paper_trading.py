import os
import yaml
import importlib.util
import torch
import torch.nn as nn
import time
from datetime import datetime, timedelta, timezone
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from binance.client import Client
import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
SCALER_DIR = BASE_DIR / "../../scaler"
SCALER_DIR = SCALER_DIR.resolve()

scaler = joblib.load(SCALER_DIR / "scaler.joblib")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
FEATURES_PY_PATH = os.path.join(EXP_DIR, "scripts", "03_pre_split_prep", "features.py")

CONF_DIR = BASE_DIR / "../../conf"
CONF_DIR = CONF_DIR.resolve()

params = yaml.safe_load(open(CONF_DIR / "params.yaml"))
keys = yaml.safe_load(open(CONF_DIR / "keys.yaml"))

spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)
else:
    raise RuntimeError(f"Could not load features.py from {FEATURES_PY_PATH}")

generate_features = getattr(features_module, "generate_features")

symbols = params["DATA_ACQUISITION"]["SYMBOLS"]
#Data Prep params
ema_periods = params['DATA_PREP']['EMA_PERIODS']
return_periods = params['DATA_PREP']['RETURN_PERIODS']
rsi_atr_window = params['DATA_PREP']['RSI_ATR_WINDOW']

#Data Model params
ROOT_DIR = Path(__file__).resolve().parents[2]
model_dir = ROOT_DIR / "model"
model_dir.mkdir(parents=True, exist_ok=True)
seq = params['MODELING']['SEQUENCE']
input_size = params['MODELING']['INPUT_SIZE']

#Api keya
api_key_id = keys['KEYS']['APCA-API-KEY-ID-Data']
api_secret = keys['KEYS']['APCA-API-SECRET-KEY-Data']
alpaca_base = "https://paper-api.alpaca.markets"

#Nutzung von GPU oder CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

net_trade = Net(input_size).to(DEVICE)
#Modellgewichte laden
net_trade.load_state_dict(torch.load(f"{model_dir}/best_model.pt"))
net_trade.eval()

trading_client = TradingClient(api_key_id, api_secret, paper=True)

def setBuyOrder(up_count, buying_count, buying_power):

    if up_count >= 12:
        buy_pct = 0.20
    elif up_count >= 8:
        buy_pct = 0.15
    elif up_count >= 5:
        buy_pct = 0.10
    else:
        buy_pct = 0.05

    notional = round(buying_power * buy_pct, 2)
    if buying_count < 11:  # UP - KAUFEN
        order_data = MarketOrderRequest(
            symbol="BTCUSD",
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.IOC
        )
        buying_count += 1
        print(f"🟢 BUY ORDER SET | buy_pct={buy_pct:.2f} | notional=${notional:.2f}")

        order = trading_client.submit_order(order_data=order_data)

        return {
            "status": order.status,
            "id": order.id,
            "side": "BUY",
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price
        }, buying_count

    else:
        print("BUYING SIGNAL -- BUT BOUGHT ALREADY 10 TIMES")
        return None, buying_count


def setSellOrder():
    try:
        position = trading_client.get_open_position("BTCUSD")
    except Exception:
        print("No open position to sell")
        return None

    qty = float(position.qty)

    if qty <= 0:
        print("Position quantity is zero")
        return None

    order_data = MarketOrderRequest(
        symbol="BTCUSD",
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.IOC
    )

    order = trading_client.submit_order(order_data=order_data)
    print(f"🟢 SELL ORDER SET | selled={qty:.2f}")
    return {
        "status": order.status,
        "id": order.id,
        "side": "SELL",
        "filled_qty": order.filled_qty,
        "filled_avg_price": order.filled_avg_price
    }


def download_and_merge_btc_eth(lookback: int) -> pd.DataFrame:
    client = Client(api_key=None, api_secret=None)

    def download_binance_last_minutes(symbol, minutes=lookback):
        klines = []

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(minutes=minutes)

        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        while start_ts < end_ts:
            data = client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                startTime=start_ts,
                endTime=end_ts,
                limit=1000
            )

            if not data:
                break

            klines.extend(data)

            # nächster Start = letzte Kerze + 1 Minute
            start_ts = data[-1][0] + 60_000

        df = pd.DataFrame(klines, columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore"
        ])

        # Zeitstempel & Datentypen
        df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        return df

    btc_df = download_binance_last_minutes(symbols[0])
    eth_df = download_binance_last_minutes(symbols[1])

    # ETH nur Close
    eth_close = eth_df[["close"]].rename(columns={"close": "eth_close"})

    merged_df = btc_df.join(eth_close, how="inner")
    merged_df.sort_index(inplace=True)
    return merged_df


def set_features(df):
    data_with_features, _ = generate_features(df, return_periods, ema_periods, rsi_atr_window)
    data_complete = data_with_features.dropna().reset_index(drop=True)

    expected_columns = list(scaler.feature_names_in_)
    data_complete = data_complete[expected_columns]
    X_scaled = scaler.transform(data_complete)
    X_scaled_df = pd.DataFrame(X_scaled, columns=data_complete.columns)

    columns_to_drop = ['open', 'high', 'low', 'eth_close', 'eth_return_5min', 'eth_return_15min', 'eth_return_60min','eth_return_90min']
    X = X_scaled_df.drop(columns=columns_to_drop, errors='ignore')

    return pd.DataFrame(X, columns=X.columns)


def predict_signal(model, x_batch):
    with torch.no_grad():
        outputs = model(x_batch)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    preds_list = preds.cpu().tolist()
    down_count = preds_list.count(0)
    hold_count = preds_list.count(1)
    up_count = preds_list.count(2)
    counts = {0: down_count, 1: hold_count, 2: up_count}
    max_class = max(counts, key=counts.get)
    print("Counts:", counts)
    return max_class, counts


def run_trading_step(holding_count, buying_count):
    raw_data = download_and_merge_btc_eth(lookback=200)
    features_df = set_features(raw_data)

    # Wir brauchen: seq für das Fenster + 15 Endpunkte
    needed = seq + 15 - 1
    if len(features_df) < needed:
        print(f"Not enough data yet. Need {needed}, have {len(features_df)}")
        return

    # Letzte 15 Minuten -> End-Indizes
    end_indices = range(len(features_df) - 15, len(features_df))  # 15 Stück

    windows = []
    for end_idx in end_indices:
        start_idx = end_idx - seq + 1
        x_window = features_df.iloc[start_idx:end_idx + 1].values  # (seq, n_features)
        windows.append(x_window)

    # Batch Tensor: (15, seq, n_features)
    x_batch = torch.tensor(np.stack(windows), dtype=torch.float32).to(DEVICE)

    max_class, counts = predict_signal(net_trade, x_batch)

    if (max_class == 2) or (holding_count > 10 and counts[0] < counts[2]):  # 0=Down, 2=Up

        account = trading_client.get_account()
        if account.trading_blocked:
            print("Account is currently restricted from trading.")
            return

        print("🟢 BUY signal")
        buying_power = float(trading_client.get_account().cash)
        if buying_power > 100:
            order, buying_count = setBuyOrder(counts[2], buying_count, buying_power)
            holding_count = 0
        else:
            print("Not enough buying power.")


    elif (max_class == 0) or (holding_count > 10 and counts[0] > counts[2]):
        print("🔴 SELL signal")
        order = setSellOrder()
        holding_count = 0
        buying_count = 0

    else:
        print("---HOLD---")
        holding_count += 1

    return holding_count, buying_count

# -----------------------------
# Main flow
# -----------------------------

last_trade_slot = None
holding_count = 0
buying_count = 0
while True:
    now = datetime.now(timezone.utc)

    # 15-Minuten-Slot bestimmen
    minute_slot = (now.minute // 15) * 15
    current_slot = now.replace(minute=minute_slot, second=0, microsecond=0)

    # Nur einmal pro 15-Minuten-Slot ausführen
    if last_trade_slot != current_slot:
        last_trade_slot = current_slot
        print(f"\nRunning trading step at {now} (slot {current_slot})")
        holding_count, buying_count = run_trading_step(holding_count, buying_count)

    time.sleep(30)

