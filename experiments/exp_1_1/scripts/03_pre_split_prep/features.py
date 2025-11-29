import pandas as pd
import numpy as np

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt 1h- und 6h-Returns für BTC und ETH hinzu."""
    df = df.copy()
    # BTC Returns
    df["btc_return_1h"] = df["close"].pct_change(periods=1)
    df["btc_return_6h"] = df["close"].pct_change(periods=6)
    df["btc_return_24h"] = df["close"].pct_change(periods=24)

    # ETH Returns
    df["eth_return_1h"] = df["eth_close"].pct_change(periods=1)
    df["eth_return_6h"] = df["eth_close"].pct_change(periods=6)
    df["eth_return_24h"] = df["eth_close"].pct_change(periods=24)
    return df

def add_eth_btc_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """ETH/BTC Ratio als relative Stärke."""
    df = df.copy()
    df["eth_btc_ratio"] = df["eth_close"] / df["close"]
    return df

def add_ema(df: pd.DataFrame, col: str, span: int, new_name: str) -> pd.DataFrame:
    """EMA-Funktion"""
    df = df.copy()
    df[new_name] = df[col].ewm(span=span, adjust=False).mean()
    return df

def add_rsi(df: pd.DataFrame, col: str, window: int = 14) -> pd.DataFrame:
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

def add_atr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["close"].shift(1)

    # True Range
    true_range = np.maximum.reduce([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ])

    # ATR
    atr_abs = pd.Series(true_range).rolling(window=window).mean()
    df[f"atr_{window}"] = atr_abs / df["close"]

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hauptfunktion: ruft alle Feature-Schritte auf"""
    df = df.copy()
    df = df.drop(columns=["trade_count"], errors="ignore")
    df = add_returns(df)
    df = add_eth_btc_ratio(df)

    # EMAs auf BTC
    df = add_ema(df, col="close", span=6, new_name="ema_6")
    df = add_ema(df, col="close", span=24, new_name="ema_24")

    # RSI
    df = add_rsi(df, col="close", window=24)

    # ATR
    df = add_atr(df, window=24)

    return df