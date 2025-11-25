from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pandas as pd
import yaml

# Load API credentials from YAML configuration file
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']

# Load data acquisition parameters from YAML configuration file
params = yaml.safe_load(open("../../conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITION']['DATA_PATH']
start_date = datetime.strptime(params['DATA_ACQUISITION']['START_DATE'], "%Y-%m-%d")
end_date = datetime.strptime(params['DATA_ACQUISITION']['END_DATE'], "%Y-%m-%d")
symbols = params["DATA_ACQUISITION"]["SYMBOLS"]

# Initialize the Alpaca client with API credentials
crypto_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

for symbol in symbols:
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date
    )

    bars = crypto_client.get_crypto_bars(request)
    df = bars.df
    df.reset_index(inplace=True)
    df.drop(columns=['symbol'], inplace=True)

    # Save the DataFrame as a Parquet file for efficient storage
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
