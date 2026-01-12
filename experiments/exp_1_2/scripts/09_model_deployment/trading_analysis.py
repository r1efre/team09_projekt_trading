import yaml
from alpaca.trading.enums import OrderSide
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
    2026, 1, 5, 0, 0, 0,
    tzinfo=timezone.utc
)

req = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    after=after_time
)

#Liste mit Orders
orders = trading_client.get_orders(filter=req)

#========================================================

berlin_tz = pytz.timezone('Europe/Berlin')


start_date = berlin_tz.localize(datetime(2026, 1, 5, 0, 0, 0))
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


# --- Kombi-Plot: Equity (orange) + BTC Close (blau) ---
fig, ax1 = plt.subplots(figsize=(12, 5))

# 1) Equity (links)
ax1.plot(x, y, color="red", label="Equity ($)")
ax1.set_xlabel("Zeit (Berlin)")
ax1.set_ylabel("Equity ($)")
ax1.grid(True)

# 2) BTC Close (rechts) - auf zweiter Achse
ax2 = ax1.twinx()

# btc_close hat Berlin-Zeitzone im Index und "Close" als Spalte
ax2.plot(btc_close.index, btc_close["Close"].astype(float), color="blue", label="BTC Close ($)")
ax2.set_ylabel("BTC Close ($)")

# Titel
plt.title("Equity vs. BTC Close (Europe/Berlin)")

# Gemeinsame Legende (beide Achsen)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.show()

if len(history.equity) > 0:
    print(f"\n{'='*70}")
    print(f"Start Portfolio: ${100000:,.2f}")
    print(f"End Portfolio: ${history.equity[-1]:,.2f}")
    change = history.equity[-1] - 100000
    change_pct = (change / 100000) * 100
    print(f"Veränderung: ${change:+,.2f} ({change_pct:+.2f}%)")

if len(btc_close) > 0:
    btc_start = float(btc_close["Close"].iloc[0])
    btc_end = float(btc_close["Close"].iloc[-1])

    btc_change = btc_end - btc_start
    btc_change_pct = (btc_change / btc_start) * 100

    print(f"\n{'='*70}")
    print(f"Start BTC Close: ${btc_start:,.2f}")
    print(f"End BTC Close:   ${btc_end:,.2f}")
    print(f"Veränderung BTC: ${btc_change:+,.2f} ({btc_change_pct:+.2f}%)")
