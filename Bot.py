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

# Vi holder det simpelt: kun ETHUSDC for lille konto
SYMBOLS = ["ETHUSDC"]

# Mere aggressivt: 1-minut candles
INTERVAL = "1m"
KLINE_LIMIT = 200

# Risiko-indstillinger
# 1.0 = brug 100% af USDC pr. BUY (giver mening ved lille saldo ~10–50 USDC)
MAX_POSITION_PCT = 1.0

# Indikator-parametre
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14

BB_PERIOD = 20
BB_STD = 2.0


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
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            # Vis Binances rå fejltekst – fx LOT_SIZE, MIN_NOTIONAL osv.
            log(f"[ORDRE FEJL HTTP] {r.status_code} {r.text}")
        except Exception:
            log(f"[ORDRE FEJL HTTP] {e}")
        raise
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
# KONTO & EXCHANGE INFO
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


def get_min_notional(symbol: str) -> float:
    """Hent MIN_NOTIONAL (mindste ordre-værdi i quote-asset, fx USDC)."""
    info = get_symbol_info(symbol)
    for f in info["filters"]:
        if f["filterType"] == "MIN_NOTIONAL":
            try:
                return float(f["minNotional"])
            except Exception:
                break
    # fallback hvis noget går galt
    return 5.0


def get_lot_step(symbol: str) -> float:
    """Hent LOT_SIZE stepSize, så vi overholder min. mængde-step."""
    info = get_symbol_info(symbol)
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            try:
                return float(f["stepSize"])
            except Exception:
                break
    # fallback
    return 0.000001


def normalize_quantity(symbol: str, qty: float) -> float:
    """
    Runder mængden NED til nærmeste gyldige LOT_SIZE-step,
    så vi ikke rammer 'Filter failure: LOT_SIZE'.
    """
    if qty <= 0:
        return 0.0
    step_size = get_lot_step(symbol)
    if step_size <= 0:
        return qty

    precision = max(0, int(round(-math.log10(step_size))))
    normalized = math.floor(qty * (10 ** precision)) / (10 ** precision)
    return normalized


# ============================
# ORDRE-HÅNDTERING
# ============================

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

    # BUY: fast over slow og RSI > 50
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


# ============================
# HOVEDLOOP
# ============================

def main_loop():
    if not API_KEY or not API_SECRET:
        log("FEJL: BINANCE_API_KEY eller BINANCE_API_SECRET mangler i .env")
        return

    log("Starter Tri-Core trading-bot (ETHUSDC, aggressiv 1m)...")
    log(f"DRY_RUN = {DRY_RUN}")
    usdc_now = get_balance("USDC")
    log(f"Start USDC balance: {usdc_now:.2f}")

    while True:
        try:
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

                if final_signal is None:
                    log(f"Ingen klar handling for {symbol}.")
                    continue

                log(f"Signal for {symbol}: {final_signal} | Årsager: {', '.join(reasons) if reasons else 'Ingen detaljer'}")

                price = get_price(symbol)
                min_notional = get_min_notional(symbol)

                if final_signal == "BUY":
                    usdc_now = get_balance("USDC")
                    usdc_for_trade = usdc_now * MAX_POSITION_PCT

                    if usdc_for_trade < min_notional:
                        log(f"For lille USDC-beløb til BUY (USDC {usdc_for_trade:.4f} < MIN_NOTIONAL {min_notional}). Skipper.")
                        continue

                    qty = usdc_for_trade / price
                    qty = normalize_quantity(symbol, qty)

                    if qty <= 0:
                        log("Normaliseret BUY-mængde <= 0. Skipper.")
                        continue

                    place_order(symbol, "BUY", qty)

                elif final_signal == "SELL":
                    base_asset = symbol.replace("USDC", "")
                    balance_base = get_balance(base_asset)
                    total_value = balance_base * price

                    if total_value < min_notional:
                        log(f"Position ({base_asset}) for lille til SELL (værdi {total_value:.4f} < MIN_NOTIONAL {min_notional}). Skipper.")
                        continue

                    qty = normalize_quantity(symbol, balance_base)

                    if qty <= 0:
                        log("Normaliseret SELL-mængde <= 0. Skipper.")
                        continue

                    place_order(symbol, "SELL", qty)

            log("Venter 60 sekunder før næste scanning...")
            time.sleep(60)

        except Exception as e:
            log(f"[FEJL i loop] {e}")
            time.sleep(10)


if __name__ == "__main__":
    main_loop()
