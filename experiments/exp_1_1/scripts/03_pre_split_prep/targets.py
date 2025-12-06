import pandas as pd
import numpy as np

#Wenn die Bewegung in der nächsten Stunde kleiner ist als 25% der typischen Tagesvolatilität, ist sie unbedeutend.
def classify_trend(row, vol_factor):
    r = row["return_1h_shifted"]
    vol = row["atr_24"]

    if pd.isna(r) or pd.isna(vol):
        return np.nan

    threshold = vol_factor * vol

    if r > threshold:
        return 2       # UP
    elif r < -threshold:
        return 0      # DOWN
    else:
        return 1      # Neutral



def set_target(df: pd.DataFrame, vol_factor=0.25) -> pd.DataFrame:
    df = df.copy()

    # Forward-Return aus return_1h
    df["return_1h_shifted"] = df["btc_return_1h"].shift(-1)

    # Klassifikation pro Zeile
    df["trend"] = df.apply(
        lambda row: classify_trend(row, vol_factor),
        axis=1
    )
    df = df.drop(columns=["return_1h_shifted"], errors="ignore")
    return df
