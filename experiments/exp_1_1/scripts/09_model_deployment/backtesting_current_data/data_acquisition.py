from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pandas as pd
import yaml

# Load API credentials (Alpaca) from YAML config
keys = yaml.safe_load(open("../../../conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']

# Load data acquisition parameters
params = yaml.safe_load(open("../../../conf/params.yaml"))
PATH_BARS = params['BACKTESTING_RECENT']['DATA_PATH_RECENT']

# Define time window for recent backtesting data
start_date = datetime.strptime("2025-12-15", "%Y-%m-%d")
end_date = datetime.strptime("2025-12-27", "%Y-%m-%d")

# Initialize Alpaca historical crypto data client
client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

# Define request for hourly OHLCV crypto bars
request_params = CryptoBarsRequest(
    symbol_or_symbols=["BTC/USD", "ETH/USD"],
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date
)

# Fetch historical crypto bars from Alpaca API
bars = client.get_crypto_bars(request_params).df.reset_index()

# Ensure timestamp column exists and is in datetime format
if "timestamp" not in bars.columns and "time" in bars.columns:
    bars = bars.rename(columns={"time": "timestamp"})
bars["timestamp"] = pd.to_datetime(bars["timestamp"])

# Save BTC and ETH data separately to parquet files
for symbol in ["BTC/USD", "ETH/USD"]:
    df = bars[bars["symbol"] == symbol].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    clean_symbol = symbol.split('/')[0]
    df.to_parquet(f'{PATH_BARS}/{clean_symbol}.parquet', index=False)

# Extract the ETH close price
eth = pd.read_parquet(f'{PATH_BARS}/ETH.parquet')
eth = eth[['timestamp', 'close']]
eth = eth.rename(columns={'close': 'eth_close'})

btc = pd.read_parquet(f'{PATH_BARS}/BTC.parquet')

# Merge BTC table with eth_close
df_merged = btc.merge(eth, on='timestamp', how='inner')
df_merged.to_parquet(f'{PATH_BARS}/dataMerged.parquet', index=False)
