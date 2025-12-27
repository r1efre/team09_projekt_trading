from datetime import datetime
from datetime import timezone
from binance.client import Client
import pandas as pd
import yaml

# Load data acquisition parameters from YAML configuration file
params = yaml.safe_load(open("../../conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITION']['DATA_PATH']
start_date = datetime.strptime(params['DATA_ACQUISITION']['START_DATE'], "%Y-%m-%d")
end_date = datetime.strptime(params['DATA_ACQUISITION']['END_DATE'], "%Y-%m-%d")
symbols = params["DATA_ACQUISITION"]["SYMBOLS"]

# Binance Client (API-Key optional für historische Daten)
client = Client(api_key=None, api_secret=None)


def download_binance_minute_data(symbol, start_date, end_date):
    klines = []

    start_date = start_date.replace(tzinfo=timezone.utc)
    end_date = end_date.replace(tzinfo=timezone.utc)
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

    # Datentypen & Index
    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("timestamp", inplace=True)

    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df

btc_df = download_binance_minute_data(symbols[0], start_date, end_date)
eth_df = download_binance_minute_data(symbols[1], start_date, end_date)

# ETH nur Close
eth_close = eth_df[["close"]].rename(columns={"close": "eth_close"})

merged_df = btc_df.join(eth_close, how="inner")
merged_df.sort_index(inplace=True)

merged_df.to_parquet(f'{PATH_BARS}/dataMerged.parquet', index=True)


