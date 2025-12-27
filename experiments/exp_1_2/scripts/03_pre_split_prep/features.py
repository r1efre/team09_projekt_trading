import pandas as pd
import numpy as np

def add_returns(df: pd.DataFrame, return_periods: list[int], features: list) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    for period in return_periods:
        df[f'btc_return_{period}min'] = df["close"].pct_change(periods=period)
        features.append(f'btc_return_{period}min')

    for period in return_periods:
        df[f'eth_return_{period}min'] = df["eth_close"].pct_change(periods=period)
        features.append(f'eth_return_{period}min')

    return df, features

def add_eth_btc_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eth_btc_ratio"] = df["eth_close"] / df["close"]
    return df

def add_ema(df: pd.DataFrame, col: str, ema_periods: list[int], features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """EMA-Funktion"""
    df = df.copy()
    for period in ema_periods:
        df[f'ema_{period}'] = df[col].ewm(span=period, adjust=False).mean()
        features.append(f'ema_{period}')

    return df, features


def add_rsi(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """RSI auf Basis von Schlusskursen"""
    df = df.copy()
    delta = df[col].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["close"].shift(1)

    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["high"] - prev_close).abs().to_numpy(),
        (df["low"] - prev_close).abs().to_numpy()
    ])

    tr = pd.Series(tr, index=df.index)

    atr_abs = tr.rolling(window=window, min_periods=window).mean()
    df["atr"] = atr_abs / df["close"]

    return df


def generate_features(df: pd.DataFrame, return_periods: list[int], ema_periods: list[int], rsi_atr_window: int) -> tuple[pd.DataFrame, list[str]]:
    """Hauptfunktion: ruft alle Feature-Schritte auf"""
    df = df.copy()
    df = df.drop(columns=["trade_count"], errors="ignore")

    features = []
    df, features = add_returns(df, return_periods, features)
    df = add_eth_btc_ratio(df)
    features.append('eth_btc_ratio')

    # EMAs auf BTC
    df, features = add_ema(df, col="close", ema_periods=ema_periods, features=features)

    # RSI
    df = add_rsi(df, col="close", window=rsi_atr_window)
    features.append('rsi')
    # ATR
    df = add_atr(df, window=rsi_atr_window)
    features.append('atr')

    return df, features