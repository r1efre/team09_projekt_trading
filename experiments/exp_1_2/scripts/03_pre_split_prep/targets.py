import pandas as pd
import numpy as np


def set_target(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    df = df.copy()

    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["trend"] = np.where(future_return > 0, 1, 0).astype(float)

    df.loc[future_return.isna(), "trend"] = np.nan

    return df


