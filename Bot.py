import os
import time
import hmac
import math
import hashlib
from urllib.parse import urlencode
from datetime import datetime

import requests
from dotenv import load_dotenv

# ============================
# KONFIGURATION
# ============================

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BASE_URL = os.getenv("BASE_URL", "https://api.binance.com")

# Sæt til True for kun at simulere (ingen rigtige handler)
# Sæt til False for rigtige handler
DRY_RUN = False

# Symboler vi handler (USDC-par)
SYMBOLS = ["BTCUSDC", "ETHUSDC", "XRPUSDC"]

# Mere aggressivt: 1-minut candles
INTERVAL = "1m"
KLINE_LIMIT = 200

# Risiko-indstillinger
MAX_POSITION_PCT = 0.5      # max 50% af konto i en trade (aggressiv for lille saldo)
DAILY_MAX_LOSS_PCT = 0.04   # 4% dagligt max tab

# Indikator-parametre
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14

BB_PERIOD = 20
BB_STD = 2.0

XRP_SYMBOL = "XRPUSDC"
XRP_SWING_RSI_LOW = 35
XRP_SWING_RSI_HIGH = 65


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


# ============================
# BINANCE API
# ============================

def _timestamp():
    return int(time.time() * 1000)


def _sign(params: dict) -> str:
    query_string = urlencode(params, True)
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return query_string + "&signature=" + signature


def _headers():
    return {"X-MBX-APIKEY": API_KEY}


def public_get(path: str, params: dict = None):
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def signed_get(path: str, params: dict = None):
    if params is None:
        params = {}
    params["timestamp"] = _timestamp()
    query = _sign(params)
    url = f"{BASE_URL}{path}?{query}"
    r = requests.get(url, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def signed_post(path: str, params: dict):
    params["timestamp"] = _timestamp()
    query = _sign(params)
    url = f"{BASE_URL}{path}"
    r = requests.post(url, headers=_headers(), data=query, timeout=10)
    r.raise_for_status()
    return r.json()


# ============================
# DATA & INDIKATORER (uden pandas)
# ============================

def fetch_klines(symbol: str, interval: str, limit: int = 200):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = public_get("/api/v3/klines", params)
    closes = [float(c[4]) for c in data]
    highs = [float(c[2]) for c in data]
    lows = [float(c[3]) for c in data]
    return closes, highs, lows


def ema(values, period):
    if len(values) < period:
        return None
    ema_values = []
    k = 2 / (period + 1)
    sma = sum(values[:period]) / period
    ema_values.append(sma)
    for price in values[period:]:
        prev = ema_values[-1]
        ema_values.append(price * k + prev * (1 - k))
    return ema_values


def rsi(values, period=14):
    if len(values) <= period:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def bollinger_last(values, period=20, std_factor=2.0):
    if len(values) < period:
        return None, None, None
    window = values[-period:]
    sma = sum(window) / period
    var = sum((x - sma) ** 2 for x in window) / period
    std = math.sqrt(var)
    upper = sma + std_factor * std
    lower = sma - std_factor * std
    return sma, upper, lower


# ============================
# KONTO & BALANCER
# ============================

def get_account():
    return signed_get("/api/v3/account")


def get_balance(asset: str) -> float:
    acc = get_account()
    for b in acc.get("balances", []):
        if b["asset"] == asset:
            return float(b["free"])
    return 0.0


def get_price(symbol: str) -> float:
    data = public_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


def get_symbol_info(symbol: str):
    data = public_get("/api/v3/exchangeInfo", {"symbol": symbol})
    return data["symbols"][0]


# ============================
# ORDRE-HÅNDTERING
# ============================

def calc_order_quantity(symbol: str, price: float, usdc_to_use: float) -> float:
    info = get_symbol_info(symbol)
    step_size = None
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
            break
    if step_size is None:
        return usdc_to_use / price

    raw_qty = usdc_to_use / price
    if step_size <= 0:
        return raw_qty
    precision = max(0, int(round(-math.log10(step_size))))
    qty = math.floor(raw_qty * (10 ** precision)) / (10 ** precision)
    return qty


def place_order(symbol: str, side: str, quantity: float, order_type: str = "MARKET"):
    if quantity <= 0:
        log(f"[ORDRE] Mængde <= 0, skipper: {symbol}")
        return None

    if DRY_RUN:
        log(f"[DRY RUN] {side} {quantity} {symbol} ({order_type})")
        return {"dry_run": True, "side": side, "symbol": symbol, "qty": quantity}

    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": f"{quantity:.8f}",
    }
    try:
        res = signed_post("/api/v3/order", params)
        log(f"[ORDRE SENDT] {side} {quantity} {symbol} → {res.get('status')}")
        return res
    except Exception as e:
        log(f"[ORDRE FEJL] {e}")
        return None


# ============================
# STRATEGI-MODULER
# ============================

def evaluate_trend(closes):
    """Modul 1: EMA 20/50 + RSI-filter (lidt løsere)."""
    if len(closes) < max(EMA_FAST, EMA_SLOW) + 2:
        return None, {}

    ema_fast_vals = ema(closes, EMA_FAST)
    ema_slow_vals = ema(closes, EMA_SLOW)

    ema_fast_last = ema_fast_vals[-1]
    ema_fast_prev = ema_fast_vals[-2]

    ema_slow_last = ema_slow_vals[-1]
    ema_slow_prev = ema_slow_vals[-2]

    rsi_last = rsi(closes, RSI_PERIOD)

    signal = None
    reason = []

    # BUY: fast over slow og RSI > 50 (ikke så strengt som før)
    if ema_fast_last > ema_slow_last and rsi_last is not None and rsi_last > 50:
        signal = "BUY"
        reason.append("EMA20 > EMA50 + RSI>50")

    # SELL: fast under slow og RSI < 50
    if ema_fast_last < ema_slow_last and rsi_last is not None and rsi_last < 50:
        signal = "SELL"
        reason.append("EMA20 < EMA50 + RSI<50")

    return signal, {
        "ema_fast_last": ema_fast_last,
        "ema_slow_last": ema_slow_last,
        "rsi_last": rsi_last,
        "reason": reason,
    }


def evaluate_scalper(closes):
    """Modul 2: Bollinger mean reversion."""
    sma, upper, lower = bollinger_last(closes, BB_PERIOD, BB_STD)
    if sma is None:
        return None, {}

    last_price = closes[-1]
    signal = None
    reason = []

    if last_price < lower:
        signal = "BUY"
        reason.append("Pris under lower Bollinger band")

    elif last_price > upper:
        signal = "SELL"
        reason.append("Pris over upper Bollinger band")

    return signal, {
        "bb_mid": sma,
        "bb_upper": upper,
        "bb_lower": lower,
        "last_price": last_price,
        "reason": reason,
    }


def evaluate_xrp_swing(closes):
    """Modul 3: XRP RSI swing."""
    if len(closes) < RSI_PERIOD + 2:
        return None, {}

    rsi_val = rsi(closes, RSI_PERIOD)
    rsi_prev = rsi(closes[:-1], RSI_PERIOD)

    signal = None
    reason = []

    if rsi_prev is not None and rsi_val is not None:
        if rsi_prev < XRP_SWING_RSI_LOW and rsi_val > rsi_prev:
            signal = "BUY"
            reason.append("RSI kommer op fra oversolgt område for XRP")
        if rsi_prev > XRP_SWING_RSI_HIGH and rsi_val < rsi_prev:
            signal = "SELL"
            reason.append("RSI falder fra høj zone for XRP")

    return signal, {
        "rsi_last": rsi_val,
        "rsi_prev": rsi_prev,
        "reason": reason,
    }


# ============================
# HOVEDLOOP
# ============================

def main_loop():
    if not API_KEY or not API_SECRET:
        log("FEJL: BINANCE_API_KEY eller BINANCE_API_SECRET mangler i .env")
        return

    log("Starter Tri-Core trading-bot (USDC-version, aggressiv 1m)...")
    log(f"DRY_RUN = {DRY_RUN}")
    usdc_start = get_balance("USDC")
    log(f"Start USDC balance: {usdc_start:.2f}")
    daily_loss_limit = usdc_start * DAILY_MAX_LOSS_PCT

    # Evt. auto-første trade: køb lidt BTC hvis vi intet ejer
    btc_bal = get_balance("BTC")
    eth_bal = get_balance("ETH")
    xrp_bal = get_balance("XRP")
    usdc_now = get_balance("USDC")

    if btc_bal == 0 and eth_bal == 0 and xrp_bal == 0 and usdc_now >= 5:
        log("Ingen BTC/ETH/XRP fundet – køber lille startposition i BTCUSDC.")
        price_btc = get_price("BTCUSDC")
        usdc_for_init = usdc_now * 0.3  # 30% af saldo til start
        qty_btc = calc_order_quantity("BTCUSDC", price_btc, usdc_for_init)
        place_order("BTCUSDC", "BUY", qty_btc)

    while True:
        try:
            usdc_now = get_balance("USDC")
            current_loss = usdc_start - usdc_now

            if current_loss > daily_loss_limit and not DRY_RUN:
                log(f"[STOP] Dagligt tab overskredet ({current_loss:.2f} USDC) – stopper for i dag.")
                break

            for symbol in SYMBOLS:
                log(f"--- Tjekker {symbol} ---")
                closes, highs, lows = fetch_klines(symbol, INTERVAL, KLINE_LIMIT)

                trend_signal, trend_info = evaluate_trend(closes)
                scalp_signal, scalp_info = evaluate_scalper(closes)

                final_signal = None
                reasons = []

                if trend_signal:
                    final_signal = trend_signal
                    reasons.extend(trend_info.get("reason", []))

                if scalp_signal:
                    reasons.extend(scalp_info.get("reason", []))
                    if final_signal is None:
                        final_signal = scalp_signal
                    elif final_signal == scalp_signal:
                        reasons.append("Trend + Scalper i samme retning (stærkt signal)")

                if symbol == XRP_SYMBOL:
                    xrp_signal, xrp_info = evaluate_xrp_swing(closes)
                    if xrp_signal:
                        reasons.extend(xrp_info.get("reason", []))
                        if final_signal is None:
                            final_signal = xrp_signal
                        elif final_signal == xrp_signal:
                            reasons.append("XRP swing modul bekræfter signal")

                if final_signal is None:
                    log(f"Ingen klar handling for {symbol}.")
                    continue

                log(f"Signal for {symbol}: {final_signal} | Årsager: {', '.join(reasons) if reasons else 'Ingen detaljer'}")

                # Risiko: hvor meget USDC bruger vi pr trade?
                usdc_now = get_balance("USDC")
                usdc_for_trade = usdc_now * MAX_POSITION_PCT
                price = get_price(symbol)

                # Tillad små trades ned til ca. 2 USDC
                if usdc_for_trade < 2:
                    log("For lidt USDC til en fornuftig trade. Skipper.")
                    continue

                qty = calc_order_quantity(symbol, price, usdc_for_trade)

                if final_signal == "BUY":
                    place_order(symbol, "BUY", qty)
                elif final_signal == "SELL":
                    base_asset = symbol.replace("USDC", "")
                    balance_base = get_balance(base_asset)
                    if balance_base > 0:
                        place_order(symbol, "SELL", balance_base)
                    else:
                        log(f"Ingen {base_asset} at sælge. Skipper SELL.")

            log("Venter 60 sekunder før næste scanning...")
            time.sleep(60)

        except Exception as e:
            log(f"[FEJL i loop] {e}")
            time.sleep(10)


if __name__ == "__main__":
    main_loop()
