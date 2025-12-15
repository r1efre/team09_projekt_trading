import os
import yaml
import importlib.util
import torch
import torch.nn as nn
import time
from datetime import datetime, timezone
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import yfinance as yf
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

#Modell definieren
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
    """
    Downloads recent hourly BTC and ETH data using yfinance,
    removes the currently forming candle, and merges ETH close into BTC data.
    """

    period_days = max(5, int((lookback / 24) + 2))

    def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and structure"""
        df = df.copy()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df.columns = [c.lower() for c in df.columns]

        # Datetime/Date zu timestamp umbenennen
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        elif 'index' in df.columns:
            df = df.rename(columns={'index': 'timestamp'})

        # Timestamp zu UTC konvertieren
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        else:
            raise RuntimeError(f"No datetime column found. Columns: {df.columns.tolist()}")

        return df

    # ==================== BTC ====================
    btc_df = yf.download(
        "BTC-USD",
        period=f"{period_days}d",
        interval="1h",
        auto_adjust=True,
        prepost=False,
        progress=False
    )

    btc_df = _normalize_yf_df(btc_df)  # Jetzt enthält es schon timestamp!

    # Remove the last (likely still forming) candle
    if len(btc_df) > 1:
        btc_df = btc_df.iloc[:-1]
    btc_df = btc_df.sort_values("timestamp").tail(lookback)

    print(f"📊 BTC: {len(btc_df)} rows")

    # ==================== ETH ====================
    eth_df = yf.download(
        "ETH-USD",
        period=f"{period_days}d",
        interval="1h",
        auto_adjust=True,
        prepost=False,
        progress=False
    )

    eth_df = _normalize_yf_df(eth_df)

    if len(eth_df) > 1:
        eth_df = eth_df.iloc[:-1]
    eth_df = eth_df.sort_values("timestamp").tail(lookback)

    # Keep only timestamp + eth close
    eth_df = eth_df[["timestamp", "close"]].rename(columns={"close": "eth_close"})

    print(f"📊 ETH: {len(eth_df)} rows")

    # ==================== MERGE ====================
    merged_df = (
        btc_df.merge(eth_df, on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged_df['vwap'] = 1
    if merged_df.empty:
        raise RuntimeError("Merged DF is empty. BTC/ETH timestamps did not align.")

    # Validate required columns
    required = ["timestamp", "open", "high", "low", "close", "volume", "eth_close"]
    missing = [c for c in required if c not in merged_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing}. Have: {merged_df.columns.tolist()}")

    print(f"✅ Merged: {len(merged_df)} rows")
    return merged_df


def set_features(df):
    data_with_features, _ = generate_features(df, return_periods, ema_periods, rsi_atr_window)
    data_complete = data_with_features.dropna().reset_index(drop=True)
    data_complete = data_complete.drop(columns=["timestamp", "symbol"], errors="ignore")

    expected_columns = list(scaler.feature_names_in_)
    data_complete = data_complete[expected_columns]
    X_scaled = scaler.transform(data_complete)
    X_scaled_df = pd.DataFrame(X_scaled, columns=data_complete.columns)

    columns_to_drop = ['open', 'high', 'low', 'vwap', 'eth_close']
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
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    if now.minute != 0 and last_trade_hour != current_hour:
        last_trade_hour = current_hour
        print(f"\nRunning trading step at {now}")
        run_trading_step()

    time.sleep(60)









