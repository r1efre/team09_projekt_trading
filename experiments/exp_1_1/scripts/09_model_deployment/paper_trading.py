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
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from pathlib import Path
import joblib

# Define the path to the scaler directory
art = Path("../../scaler")
scaler = joblib.load(art / "scaler.joblib")

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
            symbol="BTC/USD",
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.IOC
        )


    elif side == OrderSide.SELL:
        try:
            position = trading_client.get_open_position("BTC/USD")
        except Exception:
            print("No open position to sell")
            return None

        qty = float(position.qty)

        if qty <= 0:
            print("Position quantity is zero")
            return None

        order_data = MarketOrderRequest(
            symbol="BTC/USD",
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
    Downloads recent hourly BTC and ETH bars,
    keeps only fully closed candles,
    and merges ETH close price into BTC dataframe.
    Returns a clean DataFrame with a timestamp column (UTC).
    """

    crypto_client = CryptoHistoricalDataClient(
        api_key=api_key_id,
        secret_key=api_secret
    )

    # -------------------------
    # BTC
    # -------------------------
    btc_bars = crypto_client.get_crypto_bars(CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        timeframe=TimeFrame.Hour,
        limit=lookback + 2
    ))
    btc_df = btc_bars.df.copy().sort_index()

    # laufende Kerze entfernen + auf lookback kürzen
    btc_df = btc_df.iloc[:-1].tail(lookback)

    # Falls MultiIndex (timestamp, symbol) oder (symbol, timestamp): zu Spalten machen
    btc_df = btc_df.reset_index()

    # "symbol" ggf. rauswerfen (kommt je nach API als Spalte)
    btc_df = btc_df.drop(columns=["symbol"], errors="ignore")

    # timestamp sauber als UTC
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)

    # -------------------------
    # ETH
    # -------------------------
    eth_bars = crypto_client.get_crypto_bars(CryptoBarsRequest(
        symbol_or_symbols="ETH/USD",
        timeframe=TimeFrame.Hour,
        limit=lookback + 2
    ))
    eth_df = eth_bars.df.copy().sort_index()

    eth_df = eth_df.iloc[:-1].tail(lookback)
    eth_df = eth_df.reset_index()

    eth_df["timestamp"] = pd.to_datetime(eth_df["timestamp"], utc=True)

    # nur timestamp + close (umbenennen)
    eth_df = eth_df[["timestamp", "close"]].rename(columns={"close": "eth_close"})

    # -------------------------
    # Merge auf timestamp
    # -------------------------
    merged_df = btc_df.merge(eth_df, on="timestamp", how="inner")

    # final clean
    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

    # Optional: Safety-Check
    if merged_df.empty:
        raise RuntimeError(
            "Merge produced 0 rows. BTC/ETH timestamps do not align or no data returned."
        )

    return merged_df


def set_features(df):
    data_with_features, _ = generate_features(df, return_periods, ema_periods, rsi_atr_window)
    data_complete = data_with_features.drop(columns=["timestamp", "symbol"], errors="ignore")
    print("Ausführlich")
    print(data_complete.isna().sum())
    print(len(df))
    print(df.shape)  # (Zeilen, Spalten)
    print(df.shape[0])  # Nur Zeilen
    print(df.shape[1])
    X_scaled = scaler.transform(data_complete)
    X_scaled_df = pd.DataFrame(X_scaled, columns=data_complete.columns)

    columns_to_drop = ['open', 'high', 'low', 'vwap', 'eth_close']
    X = X_scaled_df.drop(columns=columns_to_drop, errors='ignore')

    print("\n🔍 Final feature names:")
    for i, col in enumerate(X.columns):
        print(f"Position {i}: {col}")

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

def has_position(symbol="BTC/USD") -> bool:
    try:
        trading_client.get_open_position(symbol)
        return True
    except Exception:
        return False

def run_trading_step():
    raw_data = download_and_merge_btc_eth(lookback=500)
    features_df = set_features(raw_data)
    if len(features_df) < seq:
        print("Not enough data for sequence yet")
        return

    x_seq = features_df.tail(seq).values
    # DEBUGGING: Prüfe auf NaN
    print("=" * 50)
    print(f"x_seq shape: {x_seq.shape}")
    print(f"NaN count: {np.isnan(x_seq).sum()}")
    print(f"First row: {x_seq[0]}")
    print(f"Last row: {x_seq[-1]}")
    print("=" * 50)

    # Wenn NaN vorhanden, abbrechen
    if np.isnan(x_seq).any():
        print("❌ ERROR: x_seq contains NaN values!")
        print(f"NaN positions:\n{np.argwhere(np.isnan(x_seq))}")
        return
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
                notional = buying_power * (0.05 if has_position("BTC/USD") else 0.10)
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
    #if now.minute != 0 and last_trade_hour != current_hour:
    last_trade_hour = current_hour
    print(f"\nRunning trading step at {now}")
    run_trading_step()

    time.sleep(10)









