import pandas as pd
import numpy as np


def set_target(df: pd.DataFrame, horizon: int = 60, threshold: float = 0.001) -> pd.DataFrame:
    df = df.copy()

    future_return = df["close"].shift(-horizon) / df["close"] - 1

    # 3 Klassen
    df["trend"] = np.select(
        [future_return > threshold,  # UP
         future_return < -threshold,  # DOWN
         True],  # NEUTRAL
        [2, 0, 1]
    ).astype(float)

    df.loc[future_return.isna(), "trend"] = np.nan

    return df


