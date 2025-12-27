import os
import pandas as pd
import yaml
import targets, features

params = yaml.safe_load(open("../../conf/params.yaml"))

# Unpack data paths and ensure processed data directory exists.
data_path = params['DATA_ACQUISITION']['DATA_PATH']
processed_path = params['DATA_PREP']['PROCESSED_PATH']
raw_data_file = params['DATA_PREP']['RAW_DATA_FILE']
os.makedirs(processed_path, exist_ok=True)

# Unpack relevant parameters for feature calculation.
ema_periods = params['DATA_PREP']['EMA_PERIODS']
return_periods = params['DATA_PREP']['RETURN_PERIODS']
rsi_atr_window = params['DATA_PREP']['RSI_ATR_WINDOW']

#Load data
rawDataPath = f"{data_path}/{raw_data_file}"
raw_data = pd.read_parquet(rawDataPath)


def check_missing_minutes(df: pd.DataFrame, freq="1min"):
    # sicherstellen: Index ist datetime + sortiert + eindeutig
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    start = df.index.min()
    end = df.index.max()

    # erwartete Zeitachse (Minute für Minute)
    expected = pd.date_range(start=start, end=end, freq=freq)

    # fehlende Minuten (liegen in expected, aber nicht im df)
    missing = expected.difference(df.index)

    # falls du wissen willst, wo Lücken-Blöcke sind (zusammenhängende Bereiche)
    missing_blocks = []
    if len(missing) > 0:
        m = missing.to_series()
        # neue Gruppe wenn Abstand > 1 Minute
        grp = (m.diff() != pd.Timedelta(freq)).cumsum()
        for _, g in m.groupby(grp):
            missing_blocks.append((g.iloc[0], g.iloc[-1], len(g)))

    # zusätzlicher Check: größte tatsächliche Zeitdifferenz
    max_gap = df.index.to_series().diff().max()

    return {
        "start": start,
        "end": end,
        "rows": len(df),
        "expected_rows": len(expected),
        "missing_count": len(missing),
        "missing_share": len(missing) / len(expected),
        "max_gap": max_gap,
        "missing_blocks": missing_blocks[:20],  # nur erste 20 Blöcke anzeigen
    }

result = check_missing_minutes(raw_data, freq="1min")
print(result)


# Features generieren
data_with_features, features = features.generate_features(raw_data, return_periods, ema_periods, rsi_atr_window)

# Targets generieren
data_complete = targets.set_target(data_with_features)

#NaN Werte entfernen, die durch das Feature Engineering entstanden sind
data_complete = data_complete.dropna().reset_index(drop=False)
print(data_complete.isna().sum())

data_complete.to_parquet(f'{processed_path}/dataComplete.parquet', index=False)

if not os.path.exists("features.txt"):
    with open("features.txt", "w") as f:
        for feature in features:
            f.write(f"{feature}\n")


desc = data_complete[["btc_return_5min", "btc_return_30min", "btc_return_60min", "eth_return_5min", "eth_return_30min", "eth_return_60min", "eth_btc_ratio", "ema_15", "ema_60", "rsi", "atr"]].describe()
print(desc.to_string())


