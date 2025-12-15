import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.trading.enums import QueryOrderStatus
import pytz

keys = yaml.safe_load(open("../../conf/keys.yaml"))

#Api key
api_key_id = keys['KEYS']['APCA-API-KEY-ID-Data']
api_secret = keys['KEYS']['APCA-API-SECRET-KEY-Data']

trading_client = TradingClient(api_key_id, api_secret, paper=True)

after_time = datetime(
    2025, 12, 14, 23, 0, 0,
    tzinfo=timezone.utc
)

req = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    after=after_time
)

orders = trading_client.get_orders(filter=req)

#========================================================

berlin_tz = pytz.timezone('Europe/Berlin')


start_date = berlin_tz.localize(datetime(2025, 12, 15, 10, 0, 0))
start_date_utc = start_date.astimezone(pytz.UTC)
end_date = datetime.now(tz=pytz.UTC)

# Portfolio History Request
request = GetPortfolioHistoryRequest(
    start=start_date_utc,
    end=end_date,
    timeframe="5Min",
    extended_hours=True
)

history = trading_client.get_portfolio_history(request)

x = []
y = []

for ts, equity in zip(history.timestamp, history.equity):
    # Alpaca timestamps sind Unix-seconds (UTC)
    dt_utc = datetime.fromtimestamp(ts, tz=pytz.UTC)
    dt_berlin = dt_utc.astimezone(berlin_tz)

    x.append(dt_berlin)
    y.append(float(equity))

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(x, y)
plt.title("Equity-Verlauf (Europe/Berlin)")
plt.xlabel("Zeit (Berlin)")
plt.ylabel("Equity ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

if len(history.equity) > 0:
    print(f"\n{'='*70}")
    print(f"Start Portfolio: ${history.equity[0]:,.2f}")
    print(f"End Portfolio: ${history.equity[-1]:,.2f}")
    change = history.equity[-1] - history.equity[0]
    change_pct = (change / history.equity[0]) * 100
    print(f"Veränderung: ${change:+,.2f} ({change_pct:+.2f}%)")


btc_df = yf.download(
    "BTC-USD",
    start=start_date_utc,
    end=end_date,
    interval="5m",
    auto_adjust=True,
    progress=False
)

# --- BTC Close in Berlin-Zeit ---
btc_close = btc_df[["Close"]].copy()

# yfinance index timezone sauber machen
if btc_close.index.tz is None:
    btc_close.index = btc_close.index.tz_localize("UTC")
btc_close.index = btc_close.index.tz_convert("Europe/Berlin")
btc_close = btc_close.sort_index()

# --- BUY marker sammeln (Zeit + Preis) ---
buy_times = []
buy_prices = []

for o in orders:
    # side robust prüfen (Enum oder str)
    side = str(o.side).lower()
    sym = getattr(o, "symbol", None)

    # Wir nehmen filled_at (besser als created_at)
    filled_at = getattr(o, "filled_at", None)

    # Nur BTC + BUY + wirklich gefüllt
    if sym not in ("BTCUSD", "BTC/USD"):
        continue
    if "buy" not in side:
        continue
    if filled_at is None:
        continue  # nicht gefüllt => kein Trade
    if float(getattr(o, "filled_qty", 0) or 0) <= 0:
        continue

    t = filled_at.astimezone(berlin_tz)

    # nur Zeitraum ab 15.12 10:00
    if not (start_date <= t <= end_date):
        continue

    # Preis zur nächsten 5-Min Candle
    idx = btc_close.index.get_indexer([t], method="nearest")[0]
    price = float(btc_close["Close"].iloc[idx])

    buy_times.append(t)
    buy_prices.append(price)

print(f"Found BUY fills to plot: {len(buy_times)}")
if buy_times[:3]:
    print("First 3 BUY times:", buy_times[:3])

# --- Plot ---
plt.figure(figsize=(13, 6))
plt.plot(btc_close.index, btc_close["Close"], label="BTC Close (5m)")

plt.scatter(
    buy_times,
    buy_prices,
    marker="^",
    s=120,
    label="BUY (filled)"
)

plt.title("BTC Preis mit BUY-Fills (ab 15.12. 10:00 Berlin)")
plt.xlabel("Zeit (Berlin)")
plt.ylabel("Preis ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Zeitraumgrenzen (volle Stunden)
start_cutoff = berlin_tz.localize(datetime(2025, 12, 15, 10, 0, 0))
end_cutoff   = berlin_tz.localize(datetime(2025, 12, 15, 20, 0, 0))

# 1) Equity DataFrame (Alpaca history -> Berlin-Zeit)
equity_df = pd.DataFrame({
    "timestamp": [
        datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(berlin_tz)
        for ts in history.timestamp
    ],
    "equity": [float(e) for e in history.equity],
}).set_index("timestamp").sort_index()

# 2) Auf Zeitraum begrenzen
equity_df = equity_df.loc[start_cutoff:end_cutoff]

# 3) "Volle Stunden" erzwingen:
#    - Wir mappen jeden Timestamp auf die Stundenmarke (z.B. 10:34 -> 10:00)
#    - und nehmen pro Stunde den ERSTEN Wert (nahe an der vollen Stunde)
equity_full_hours = equity_df.copy()
equity_full_hours["hour"] = equity_full_hours.index.floor("H")
equity_full_hours = (
    equity_full_hours
    .groupby("hour")["equity"]
    .first()
    .to_frame()
)

# Optional: sicherstellen, dass wirklich nur volle Stunden im Index sind
equity_full_hours = equity_full_hours.loc[start_cutoff:end_cutoff]

# 4) Änderungen von voller Stunde zu voller Stunde
equity_full_hours["delta_$"]  = equity_full_hours["equity"].diff()
equity_full_hours["return_%"] = equity_full_hours["equity"].pct_change() * 100

# Für Durchschnitt: erste Zeile entfernen
calc_df = equity_full_hours.dropna()

avg_delta  = calc_df["delta_$"].mean()
avg_return = calc_df["return_%"].mean()

# 5) Ausgabe formatieren
table = equity_full_hours.copy()
table.index = table.index.strftime("%d.%m.%Y %H:%M")

table["equity_$"] = table["equity"].map(lambda v: f"{v:,.2f}")
table["delta_$"]  = table["delta_$"].map(lambda v: "" if pd.isna(v) else f"{v:+,.2f}")
table["return_%"] = table["return_%"].map(lambda v: "" if pd.isna(v) else f"{v:+.4f}%")

display_table = table[["equity_$", "delta_$", "return_%"]]

# 6) Print
print("\nStündliche Entwicklung (volle Stunde -> volle Stunde) – 15.12.2025 10:00 bis 20:00 (Berlin):")
print(display_table.to_string())

print("\n" + "=" * 80)
print("Durchschnittliche stündliche Entwicklung (ohne erste Stunde):")
print(f"Ø Änderung pro Stunde:   ${avg_delta:+,.2f}")
print(f"Ø Return pro Stunde:     {avg_return:+.4f}%")
print("=" * 80)

