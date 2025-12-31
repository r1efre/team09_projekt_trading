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
params = yaml.safe_load(open("../../conf/params.yaml"))
keys = yaml.safe_load(open("../../conf/keys.yaml"))

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
model_path = params['MODELING']['SAVE_MODEL']
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
net_trade.load_state_dict(torch.load(f"{model_path}/best_model.pt"))
net_trade.eval()

trading_client = TradingClient(api_key_id, api_secret, paper=True)

def place_order(side, notional=None):

    if side == OrderSide.BUY:

        order_data = MarketOrderRequest(
            symbol="BTCUSD",
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.IOC
        )

    elif side == OrderSide.SELL:
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

    else:
        raise ValueError("Invalid order side")

    order = trading_client.submit_order(order_data=order_data)

    return {
        "status": order.status,
        "id": order.id,
        "side": side.name,
        "filled_qty": order.filled_qty,
        "filled_avg_price": order.filled_avg_price
    }


def download_and_merge_btc_eth(lookback: int) -> pd.DataFrame:
    client = Client(api_key=None, api_secret=None)

    def download_binance_last_minutes(symbol, minutes=120):
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


def predict_signal(model, x_seq: np.ndarray):
    x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        print(probs)

    prob_down = probs[0].item()
    prob_hold = probs[1].item()
    prob_up = probs[2].item()
    print("prob_down:", prob_down)
    print("prob_hold:", prob_hold)
    print("prob_up:", prob_up)

    predicted = torch.argmax(probs).item()
    diff = abs(prob_up - prob_down) * 100
    print(diff)

    return predicted, prob_down, prob_hold, prob_up, diff

def has_position(symbol="BTCUSD") -> bool:
    try:
        trading_client.get_open_position(symbol)
        return True
    except Exception:
        return False

def run_trading_step():
    raw_data = download_and_merge_btc_eth(lookback=120)
    features_df = set_features(raw_data)
    if len(features_df) < seq:
        print("Not enough data for sequence yet")
        return

    x_seq = features_df.tail(seq).values
    predicted, p_down, p_hold, p_up, diff = predict_signal(net_trade, x_seq)

    print(f"Prediction: {predicted} | diff={diff:.2f}")

    if predicted in [0, 2] and diff >= 5:
        account = trading_client.get_account()
        if account.trading_blocked:
            print("Account is currently restricted from trading.")
            return

        if predicted == 2:
            print("🟢 BUY signal")
            buying_power = float(trading_client.get_account().buying_power)
            if buying_power > 100:
                notional = round(buying_power * (0.05 if has_position("BTCUSD") else 0.10),2)
                place_order(OrderSide.BUY, notional=notional)
            else:
                print("Not enough buying power.")

        elif predicted == 0:
            print("🔴 SELL signal")
            place_order(OrderSide.SELL)

    else:
        print("⏸ HOLD")



# -----------------------------
# Main flow
# -----------------------------

last_trade_hour = None

while True:
    now = datetime.now(timezone.utc)

    # Slot auf 30-Minuten-Intervalle runden (00 oder 30)
    minute_slot = 0 if now.minute < 30 else 30
    current_slot = now.replace(minute=minute_slot, second=0, microsecond=0)

    # Nur einmal pro Slot handeln, sobald wir im Slot sind
    if last_trade_slot != current_slot and now.minute % 30 != 0:
        last_trade_slot = current_slot
        print(f"\nRunning trading step at {now}")
        run_trading_step()

    time.sleep(30)
