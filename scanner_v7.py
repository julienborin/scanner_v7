# -*- coding: utf-8 -*-
# ============================================================
# TRADING SCANNER v7.0 (FUSION ULTIMATE + CONNECT)
# MT5 + ML + Ichimoku + Check-list + Backtest + Simulation
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import pytz
import requests
import json
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Trading Scanner v7.0", page_icon="🧠", layout="wide")

# ══════════════════════════════════════════════════════════
# FICHIERS DE PERSISTANCE
# ══════════════════════════════════════════════════════════

CONFIG_FILE = "scanner_config.json"
SIMULATION_FILE = "simulation_data.json"
ALERTES_PRIX_FILE = "alertes_prix.json"


def charger_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            pass
    return {} if filepath == CONFIG_FILE else []


def sauver_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


config = charger_json(CONFIG_FILE)


def save_config():
    sauver_json(CONFIG_FILE, config)


# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════

if "historique_signaux" not in st.session_state:
    st.session_state.historique_signaux = []
if "mt5_connected" not in st.session_state:
    st.session_state.mt5_connected = False
if "show_mt5" not in st.session_state:
    st.session_state.show_mt5 = False
if "mt5_auto_config" not in st.session_state:
    st.session_state.mt5_auto_config = {"enabled": False}
if "derniers_resultats" not in st.session_state:
    st.session_state.derniers_resultats = []
if "scan_effectue" not in st.session_state:
    st.session_state.scan_effectue = False
if "show_sim" not in st.session_state:
    st.session_state.show_sim = False
if "dernieres_divergences" not in st.session_state:
    st.session_state.dernieres_divergences = {}

# ══════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════

ACTIFS = {
    "🥇 Or (Gold)": "GC=F",
    "🥈 Argent (Silver)": "SI=F",
    "💶 EUR/CHF": "EURCHF=X",
    "💵 EUR/USD": "EURUSD=X",
    "₿ Bitcoin": "BTC-USD",
    "🍎 Apple": "AAPL",
    "💻 Microsoft": "MSFT",
    "🚗 Tesla": "TSLA",
    "⟠ Ethereum": "ETH-USD",
    "⬜ Platine": "PL=F",
    "🛢️ Pétrole": "CL=F",
    "📊 S&P 500": "^GSPC",
    "💎 Solana": "SOL-USD",
    "🔗 Chainlink": "LINK-USD",
    "🟡 BNB": "BNB-USD",
    "📈 Nasdaq": "^IXIC",
}

CCXT_SYMBOLS = {
    "BTC-USD": "BTC/USDT", "ETH-USD": "ETH/USDT",
    "SOL-USD": "SOL/USDT", "LINK-USD": "LINK/USDT", "BNB-USD": "BNB/USDT",
}

MT5_SYMBOLS = {
    "GC=F": "GOLD", "SI=F": "SILVER", "EURCHF=X": "EURCHF",
    "EURUSD=X": "EURUSD", "BTC-USD": "BTCUSD", "AAPL": "AAPL",
    "MSFT": "MSFT", "TSLA": "TSLA", "ETH-USD": "ETHUSD",
    "PL=F": "PLATINUM", "CL=F": "OIL_CRUDE", "^GSPC": "SP500m",
    "SOL-USD": "SOLUSD", "LINK-USD": "LINKUSD", "BNB-USD": "BNBUSD",
    "^IXIC": "US_TECH100",
}

POIDS = {
    "RSI": 2.0, "MACD": 2.0, "STOCH": 1.0, "FIBO": 1.5,
    "MA200": 1.5, "VOLUME": 1.5, "BOLLINGER": 1.5, "DIVERGENCE": 2.0,
    "OR_BTC": 1.5, "MACRO": 2.5, "SENTIMENT": 2.0, "NEWS_NLP": 2.0,
    "ONCHAIN": 2.5, "ICHIMOKU": 2.0, "SUPPORTS_RES": 1.5,
    "ML_PREDICTION": 3.0, "VWAP": 1.0, "ORDER_FLOW": 1.5,
    "DIV_METALS": 2.0,
}

SCORE_MAX = sum(POIDS.values())
SEUIL_ADX = 25

ACTIF_CATEGORIE = {
    "🥇 Or (Gold)": "commodities", "🥈 Argent (Silver)": "commodities",
    "💶 EUR/CHF": "forex", "💵 EUR/USD": "forex",
    "₿ Bitcoin": "crypto", "🍎 Apple": "stocks", "💻 Microsoft": "stocks",
    "🚗 Tesla": "stocks", "⟠ Ethereum": "crypto", "⬜ Platine": "commodities",
    "🛢️ Pétrole": "commodities", "📊 S&P 500": "stocks",
    "💎 Solana": "crypto", "🔗 Chainlink": "crypto", "🟡 BNB": "crypto",
    "📈 Nasdaq": "stocks",
}

HORAIRES = {
    "forex": {"buy": [8, 9, 10, 14, 15, 16], "sell": [8, 9, 10, 14, 15, 16], "avoid": [12, 13, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5], "buy_txt": "8h-10h ou 14h-16h", "sell_txt": "8h-10h ou 14h-16h"},
    "crypto": {"buy": [6, 7, 8], "sell": [15, 16, 17], "avoid": [22, 23, 0, 1, 2], "buy_txt": "6h-8h", "sell_txt": "15h-17h"},
    "stocks": {"buy": [15, 16], "sell": [19, 20, 21], "avoid": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "buy_txt": "15h45-16h15", "sell_txt": "19h-21h"},
    "commodities": {"buy": [8, 9, 10, 14, 15], "sell": [16, 17, 18], "avoid": [21, 22, 23, 0, 1, 2, 3, 4, 5], "buy_txt": "8h-10h ou 14h30", "sell_txt": "16h-18h"},
}

NEWS_KEYWORDS = {
    "GC=F": ["gold price", "XAUUSD", "precious metals"],
    "SI=F": ["silver price", "XAGUSD"],
    "EURCHF=X": ["EUR CHF", "SNB", "swiss franc"],
    "EURUSD=X": ["EUR USD", "Fed rates", "ECB"],
    "BTC-USD": ["bitcoin", "BTC", "bitcoin ETF"],
    "ETH-USD": ["ethereum", "ETH", "DeFi"],
    "SOL-USD": ["solana", "SOL"],
    "AAPL": ["Apple stock", "AAPL", "iPhone"],
    "MSFT": ["Microsoft", "MSFT", "Azure"],
    "TSLA": ["Tesla", "TSLA", "Elon Musk"],
    "CL=F": ["crude oil", "OPEC", "WTI"],
    "^GSPC": ["S&P 500", "Wall Street"],
    "PL=F": ["platinum"],
    "LINK-USD": ["chainlink", "LINK"],
    "BNB-USD": ["BNB", "binance"],
    "^IXIC": ["nasdaq", "tech stocks"],
}

BULLISH_KW = ["rally", "surge", "breakout", "bullish", "all-time high", "inflows", "upgrade", "rate cut", "approval", "record", "accumulation", "beat expectations", "adoption", "institutional buying", "outperform"]
BEARISH_KW = ["crash", "plunge", "selloff", "bearish", "dump", "liquidation", "hack", "ban", "hawkish", "recession", "downgrade", "warning", "fear", "bankruptcy", "investigation", "lawsuit", "outflows"]

MACRO_SENSITIVITY = {"crypto": 0.9, "commodities": 0.85, "forex": 0.7, "stocks": 0.75}


# ══════════════════════════════════════════════════════════
# MT5
# ══════════════════════════════════════════════════════════

def mt5_connect(login, password, server):
    if not HAS_MT5:
        return False, "MetaTrader5 non installé"
    if not mt5.initialize():
        return False, f"Erreur init: {mt5.last_error()}"
    if not mt5.login(login=int(login), password=password, server=server):
        mt5.shutdown()
        return False, f"Login échoué: {mt5.last_error()}"
    info = mt5.account_info()
    if not info:
        mt5.shutdown()
        return False, "Infos indisponibles"
    return True, info


def mt5_disconnect():
    if HAS_MT5:
        mt5.shutdown()
    st.session_state.mt5_connected = False


def mt5_account_info():
    if not HAS_MT5 or not st.session_state.mt5_connected:
        return None
    try:
        info = mt5.account_info()
        if info:
            return {"balance": info.balance, "equity": info.equity, "profit": info.profit, "margin_free": info.margin_free, "currency": info.currency, "leverage": info.leverage, "login": info.login, "server": info.server, "mode": "DEMO" if info.trade_mode == 0 else "RÉEL"}
    except:
        pass
    return None


def mt5_open_trade(ticker, direction, lots, sl=None, tp=None):
    if not HAS_MT5 or not st.session_state.mt5_connected:
        return False, "Non connecté"
    symbol = MT5_SYMBOLS.get(ticker)
    if not symbol:
        return False, f"{ticker} non mappé"
    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        for sfx in [".ava", "_ava", ""]:
            sym_info = mt5.symbol_info(symbol + sfx)
            if sym_info:
                symbol = symbol + sfx
                break
    if not sym_info:
        return False, f"{symbol} introuvable"
    if not sym_info.visible:
        mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False, "Pas de cotation"
    if direction == "ACHAT":
        otype = mt5.ORDER_TYPE_BUY; price = tick.ask
    else:
        otype = mt5.ORDER_TYPE_SELL; price = tick.bid
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lots), "type": otype, "price": price, "deviation": 20, "magic": 700000, "comment": "ScannerV7", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    if sl and sl > 0:
        req["sl"] = float(sl)
    if tp and tp > 0:
        req["tp"] = float(tp)
    result = mt5.order_send(req)
    if not result:
        return False, f"Erreur: {mt5.last_error()}"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Rejeté [{result.retcode}]: {result.comment}"
    return True, {"ticket": result.order, "symbol": symbol, "direction": direction, "lots": lots, "price": price}


def mt5_close_trade(ticket):
    if not HAS_MT5:
        return False, "MT5 indisponible"
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False, f"#{ticket} introuvable"
    p = pos[0]
    if p.type == mt5.ORDER_TYPE_BUY:
        otype = mt5.ORDER_TYPE_SELL; price = mt5.symbol_info_tick(p.symbol).bid
    else:
        otype = mt5.ORDER_TYPE_BUY; price = mt5.symbol_info_tick(p.symbol).ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "type": otype, "position": ticket, "price": price, "deviation": 20, "magic": 700000, "comment": "V7 Close", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    result = mt5.order_send(req)
    if not result:
        return False, f"Erreur: {mt5.last_error()}"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Rejeté [{result.retcode}]"
    return True, {"ticket": ticket, "profit": p.profit}


def mt5_get_positions():
    if not HAS_MT5 or not st.session_state.mt5_connected:
        return []
    try:
        positions = mt5.positions_get()
        if not positions:
            return []
        return [{"ticket": p.ticket, "symbol": p.symbol, "type": "ACHAT" if p.type == 0 else "VENTE", "volume": p.volume, "price_open": p.price_open, "price_current": p.price_current, "sl": p.sl, "tp": p.tp, "profit": p.profit, "swap": p.swap, "time": datetime.fromtimestamp(p.time).strftime("%d.%m %H:%M")} for p in positions]
    except:
        return []
# ══════════════════════════════════════════════════════════
# TELEGRAM + EMAIL
# ══════════════════════════════════════════════════════════

def envoyer_telegram(message, token, chat_id):
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except:
        return False


def format_telegram(r):
    emoji = "🟢" if r["action"] == "ACHAT" else "🔴"
    d = "LONG" if r["action"] == "ACHAT" else "SHORT"
    sc = r["score_achat"] if r["action"] == "ACHAT" else r["score_vente"]
    msg = f"{emoji} <b>{d}</b> — {r['nom']}\n💰 {round(r['prix'], 2)} | Score: {round(sc, 1)}/{round(SCORE_MAX, 0)}"
    if r.get("sl_tp"):
        msg += f"\n🛑 SL: {round(r['sl_tp']['stop_loss'], 2)} | 🎯 TP: {round(r['sl_tp']['take_profit'], 2)} | R:R 1:{round(r['sl_tp']['ratio_rr'], 1)}"
    if r.get("ml") and r["ml"].get("acc", 0) > 0.52:
        msg += f"\n🤖 ML: {round(r['ml']['hausse'] * 100, 0)}% (acc {round(r['ml']['acc'] * 100, 0)}%)"
    if r.get("divergences_txt"):
        msg += f"\n🔀 {r['divergences_txt']}"
    msg += f"\n⏰ {datetime.now(pytz.timezone('Europe/Zurich')).strftime('%H:%M:%S')}"
    return msg


def envoyer_email(sujet, message, email_addr, email_pass):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = sujet; msg['From'] = email_addr; msg['To'] = email_addr
        msg.attach(MIMEText(message, 'plain', 'utf-8'))
        with smtplib.SMTP_SSL("smtpauths.bluewin.ch", 465) as server:
            server.login(email_addr, email_pass); server.send_message(msg)
        return True
    except Exception as e:
        return str(e)


# ══════════════════════════════════════════════════════════
# DATA — TÉLÉCHARGEMENT
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_ccxt(symbol, tf="1d", limit=365):
    if not HAS_CCXT:
        return None
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df
    except:
        return None


@st.cache_data(ttl=30)
def fetch_orderbook(symbol):
    if not HAS_CCXT:
        return None
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        ob = ex.fetch_order_book(symbol, limit=50)
        bids = sum(b[1] for b in ob["bids"][:20])
        asks = sum(a[1] for a in ob["asks"][:20])
        total = bids + asks
        if total == 0:
            return None
        return {"imbalance": (bids - asks) / total, "bids": bids, "asks": asks}
    except:
        return None


@st.cache_data(ttl=300, show_spinner="📥 Données...")
def telecharger(ticker):
    if HAS_CCXT and ticker in CCXT_SYMBOLS:
        data = fetch_ccxt(CCXT_SYMBOLS[ticker], "1d", 365)
        if data is not None and not data.empty:
            return data
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


@st.cache_data(ttl=300)
def telecharger_4h(ticker):
    if HAS_CCXT and ticker in CCXT_SYMBOLS:
        data = fetch_ccxt(CCXT_SYMBOLS[ticker], "4h", 200)
        if data is not None and not data.empty:
            return data
    data = yf.download(ticker, period="60d", interval="1h", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        return data
    return data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()


@st.cache_data(ttl=300)
def telecharger_weekly(ticker):
    data = yf.download(ticker, period="2y", interval="1wk", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def get_prix_actuel(ticker):
    try:
        if HAS_CCXT and ticker in CCXT_SYMBOLS:
            ex = ccxt.binance({"enableRateLimit": True})
            t = ex.fetch_ticker(CCXT_SYMBOLS[ticker])
            if t and t.get("last"):
                return float(t["last"])
    except:
        pass
    try:
        t = yf.Ticker(ticker)
        p = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
        if p and p > 0:
            return float(p)
    except:
        pass
    try:
        d = yf.download(ticker, period="5d", interval="1d", progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if not d.empty:
            return float(d["Close"].iloc[-1])
    except:
        pass
    return None


# ══════════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ══════════════════════════════════════════════════════════

def calc_rsi(s, p=14):
    if HAS_PANDAS_TA:
        r = ta.rsi(s, length=p)
        if r is not None:
            return r
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(p).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + gain / loss))


def calc_stoch(data, p=14):
    if HAS_PANDAS_TA:
        r = ta.stoch(data["High"], data["Low"], data["Close"], k=p)
        if r is not None and not r.empty:
            c = r.columns.tolist()
            return r[c[0]], r[c[1]]
    lo = data["Low"].rolling(p).min()
    hi = data["High"].rolling(p).max()
    k = ((data["Close"] - lo) / (hi - lo)) * 100
    return k, k.rolling(3).mean()


def calc_macd(close):
    if HAS_PANDAS_TA:
        r = ta.macd(close, fast=12, slow=26, signal=9)
        if r is not None and not r.empty:
            c = r.columns.tolist()
            return r[c[0]], r[c[2]]
    e12 = close.ewm(span=12).mean()
    e26 = close.ewm(span=26).mean()
    m = e12 - e26
    return m, m.ewm(span=9).mean()


def calc_adx(data, p=14):
    if HAS_PANDAS_TA:
        r = ta.adx(data["High"], data["Low"], data["Close"], length=p)
        if r is not None and not r.empty:
            return r.iloc[:, 0]
    h = data["High"]; l = data["Low"]; c = data["Close"]
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    pdm = h.diff().where(h.diff() > -l.diff(), 0).where(h.diff() > 0, 0)
    mdm = (-l.diff()).where(-l.diff() > h.diff(), 0).where(-l.diff() > 0, 0)
    atr = tr.rolling(p).mean()
    pdi = 100 * pdm.rolling(p).mean() / atr
    mdi = 100 * mdm.rolling(p).mean() / atr
    dx = 100 * abs(pdi - mdi) / (pdi + mdi)
    return dx.rolling(p).mean()


def calc_atr(data, p=14):
    if HAS_PANDAS_TA:
        r = ta.atr(data["High"], data["Low"], data["Close"], length=p)
        if r is not None:
            return r
    h = data["High"]; l = data["Low"]; c = data["Close"]
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.rolling(p).mean()


def calc_bollinger(close, p=20, m=2):
    if HAS_PANDAS_TA:
        r = ta.bbands(close, length=p, std=m)
        if r is not None and not r.empty:
            c = r.columns.tolist()
            return r[c[2]], r[c[0]], r[c[1]]
    sma = close.rolling(p).mean()
    std = close.rolling(p).std()
    return sma + m * std, sma - m * std, sma


def calc_ichimoku(data):
    h = data["High"]; l = data["Low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spa = ((tenkan + kijun) / 2).shift(26)
    spb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, spa, spb


def calc_vwap(data):
    if HAS_PANDAS_TA:
        r = ta.vwap(data["High"], data["Low"], data["Close"], data["Volume"])
        if r is not None:
            return r
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    return (tp * data["Volume"]).cumsum() / data["Volume"].cumsum()


def calc_fibonacci(data, p=50):
    hi = data["High"].rolling(p).max()
    lo = data["Low"].rolling(p).min()
    return hi - 0.618 * (hi - lo), hi - 0.382 * (hi - lo)


def detect_supports_resistances(data, window=20):
    close = data["Close"].values; high = data["High"].values; low = data["Low"].values
    supports = []; resistances = []
    for i in range(window, len(close) - window):
        if low[i] == min(low[i - window:i + window + 1]):
            supports.append(low[i])
        if high[i] == max(high[i - window:i + window + 1]):
            resistances.append(high[i])

    def cluster(lvls):
        if not lvls:
            return []
        lvls = sorted(lvls); cl = [lvls[0]]
        for lv in lvls[1:]:
            if (lv - cl[-1]) / cl[-1] > 0.02:
                cl.append(lv)
            else:
                cl[-1] = (cl[-1] + lv) / 2
        return cl

    supports = cluster(supports); resistances = cluster(resistances)
    px = close[-1]
    supports = sorted(supports, key=lambda x: abs(x - px))[:5]
    resistances = sorted(resistances, key=lambda x: abs(x - px))[:5]
    return sorted(supports), sorted(resistances)


def detecter_divergences_tech(data, lookback=14):
    result = {'rsi': None, 'macd': None}
    if len(data) < lookback + 5:
        return result
    close = data['Close'].values; rsi = data['RSI'].values; macd = data['MACD'].values
    try:
        recent = close[-lookback:]; recent_rsi = rsi[-lookback:]; recent_macd = macd[-lookback:]
        prix_lows = []
        for i in range(2, lookback - 2):
            if recent[i] <= min(recent[i-2:i]) and recent[i] <= min(recent[i+1:i+3]):
                prix_lows.append((i, recent[i], recent_rsi[i], recent_macd[i]))
        if len(prix_lows) >= 2:
            last = prix_lows[-1]; prev = prix_lows[-2]
            if last[1] < prev[1] and last[2] > prev[2]:
                result['rsi'] = "HAUSSIERE"
            if last[1] < prev[1] and last[3] > prev[3]:
                result['macd'] = "HAUSSIERE"
        prix_highs = []
        for i in range(2, lookback - 2):
            if recent[i] >= max(recent[i-2:i]) and recent[i] >= max(recent[i+1:i+3]):
                prix_highs.append((i, recent[i], recent_rsi[i], recent_macd[i]))
        if len(prix_highs) >= 2:
            last = prix_highs[-1]; prev = prix_highs[-2]
            if last[1] > prev[1] and last[2] < prev[2]:
                result['rsi'] = "BAISSIERE"
            if last[1] > prev[1] and last[3] < prev[3]:
                result['macd'] = "BAISSIERE"
    except:
        pass
    return result


def calculer_indicateurs(data):
    data["RSI"] = calc_rsi(data["Close"])
    data["Stoch_K"], data["Stoch_D"] = calc_stoch(data)
    data["MACD"], data["MACD_Signal"] = calc_macd(data["Close"])
    data["ADX"] = calc_adx(data)
    data["ATR"] = calc_atr(data)
    data["BB_Upper"], data["BB_Lower"], data["BB_Mid"] = calc_bollinger(data["Close"])
    data["MA_200"] = data["Close"].rolling(200).mean()
    data["MA_50"] = data["Close"].rolling(50).mean()
    data["Vol_Moy"] = data["Volume"].rolling(20).mean()
    data["VWAP"] = calc_vwap(data)
    data["Fib_618"], data["Fib_382"] = calc_fibonacci(data)
    tk, kj, spa, spb = calc_ichimoku(data)
    data["Ichi_TK"] = tk; data["Ichi_KJ"] = kj
    data["Ichi_SpA"] = spa; data["Ichi_SpB"] = spb
    return data


# ══════════════════════════════════════════════════════════
# DIVERGENCES DXY / OR / ARGENT
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner="🔀 Divergences métaux...")
def analyser_divergences_metals():
    try:
        dxy = yf.download("DX-Y.NYB", period="30d", interval="1d", progress=False)
        or_data = yf.download("GC=F", period="30d", interval="1d", progress=False)
        argent_data = yf.download("SI=F", period="30d", interval="1d", progress=False)
        for d in [dxy, or_data, argent_data]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
        if len(dxy) < 10 or len(or_data) < 10 or len(argent_data) < 10:
            return {"erreur": "Pas assez de données"}
        var_dxy = float((dxy['Close'].iloc[-1] - dxy['Close'].iloc[-10]) / dxy['Close'].iloc[-10] * 100)
        var_or = float((or_data['Close'].iloc[-1] - or_data['Close'].iloc[-10]) / or_data['Close'].iloc[-10] * 100)
        var_ag = float((argent_data['Close'].iloc[-1] - argent_data['Close'].iloc[-10]) / argent_data['Close'].iloc[-10] * 100)
        res = {}
        if var_dxy > 0 and var_or > 0:
            res["DXY_vs_Or"] = {"signal": "🟢 FORCE", "detail": f"DXY ({var_dxy:+.1f}%) + Or ({var_or:+.1f}%) → Force Or", "impact_or": "renforce_achat"}
        elif var_dxy < 0 and var_or < 0:
            res["DXY_vs_Or"] = {"signal": "🔴 FAIBLESSE", "detail": f"DXY ({var_dxy:+.1f}%) + Or ({var_or:+.1f}%) → Faiblesse", "impact_or": "renforce_vente"}
        else:
            res["DXY_vs_Or"] = {"signal": "⚪ Normal", "detail": f"DXY ({var_dxy:+.1f}%) vs Or ({var_or:+.1f}%)", "impact_or": "neutre"}
        if var_dxy > 0 and var_ag > 0:
            res["DXY_vs_Argent"] = {"signal": "🟢 FORCE", "detail": "DXY+Ag force", "impact_argent": "renforce_achat"}
        elif var_dxy < 0 and var_ag < 0:
            res["DXY_vs_Argent"] = {"signal": "🔴 FAIBLESSE", "detail": "DXY-Ag faiblesse", "impact_argent": "renforce_vente"}
        else:
            res["DXY_vs_Argent"] = {"signal": "⚪ Normal", "detail": f"DXY ({var_dxy:+.1f}%) vs Ag ({var_ag:+.1f}%)", "impact_argent": "neutre"}
        if var_or > 0 and var_ag < 0:
            res["Or_vs_Argent"] = {"signal": "🟡 PRUDENCE", "detail": "Or↑ sans Argent = peur", "impact_or": "affaiblit_achat", "impact_argent": "neutre"}
        elif var_or < 0 and var_ag > 0:
            res["Or_vs_Argent"] = {"signal": "🟢 RATTRAPAGE", "detail": "Argent rattrape", "impact_or": "neutre", "impact_argent": "renforce_achat"}
        else:
            res["Or_vs_Argent"] = {"signal": "⚪ Normal", "detail": "Même direction", "impact_or": "neutre", "impact_argent": "neutre"}
        res["variations"] = {"DXY": f"{var_dxy:+.1f}%", "Or": f"{var_or:+.1f}%", "Argent": f"{var_ag:+.1f}%"}
        return res
    except Exception as e:
        return {"erreur": str(e)}


def get_divergence_metal_impact(ticker, divergences):
    if "erreur" in divergences:
        return 0, ""
    if ticker == "GC=F":
        imp = divergences.get("DXY_vs_Or", {}).get("impact_or", "neutre")
        imp2 = divergences.get("Or_vs_Argent", {}).get("impact_or", "neutre")
        if imp == "renforce_achat":
            return 1, "DXY↑ + Or↑"
        elif imp == "renforce_vente":
            return -1, "DXY↓ + Or↓"
        elif imp2 == "affaiblit_achat":
            return -1, "Or↑ sans Ag"
        return 0, ""
    elif ticker == "SI=F":
        imp = divergences.get("DXY_vs_Argent", {}).get("impact_argent", "neutre")
        imp2 = divergences.get("Or_vs_Argent", {}).get("impact_argent", "neutre")
        if imp == "renforce_achat" or imp2 == "renforce_achat":
            return 1, "Force Argent"
        elif imp == "renforce_vente":
            return -1, "Faiblesse Ag"
        return 0, ""
    elif ticker == "PL=F":
        imp = divergences.get("DXY_vs_Argent", {}).get("impact_argent", "neutre")
        if imp == "renforce_achat":
            return 1, "Force métaux"
        elif imp == "renforce_vente":
            return -1, "Faiblesse métaux"
        return 0, ""
    return 0, ""


# ══════════════════════════════════════════════════════════
# OR/BTC DIVERGENCE
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def get_divergence_or_btc():
    try:
        g = yf.download("GC=F", period="30d", interval="1d", progress=False)
        b = yf.download("BTC-USD", period="30d", interval="1d", progress=False)
        if isinstance(g.columns, pd.MultiIndex):
            g.columns = g.columns.get_level_values(0)
        if isinstance(b.columns, pd.MultiIndex):
            b.columns = b.columns.get_level_values(0)
        if len(g) < 7 or len(b) < 7:
            return None
        vo = float((g["Close"].iloc[-1] - g["Close"].iloc[-7]) / g["Close"].iloc[-7] * 100)
        vb = float((b["Close"].iloc[-1] - b["Close"].iloc[-7]) / b["Close"].iloc[-7] * 100)
        return {"var_or": vo, "var_btc": vb, "ecart": vo - vb}
    except:
        return None


def indicateur_or_btc(ticker):
    data = get_divergence_or_btc()
    if not data:
        return 0, ""
    ecart = data["ecart"]
    if ticker == "GC=F":
        if ecart < -5:
            return 1, "BTC > Or → rattrapage"
        elif ecart > 5:
            return -1, "Or > BTC → excès"
    elif ticker in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        if ecart > 5:
            return 1, "Or > Crypto → rattrapage"
        elif ecart < -5:
            return -1, "Crypto > Or → excès"
    return 0, ""


# ══════════════════════════════════════════════════════════
# MACRO + NEWS + ON-CHAIN
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner="🌍 Macro...")
def fetch_macro():
    data = {}; details = []

    def dl(ticker):
        d = yf.download(ticker, period="60d", interval="1d", progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return d

    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if r.status_code == 200:
            fg = int(r.json()["data"][0]["value"])
            fg_label = r.json()["data"][0]["value_classification"]
            data["fear_greed"] = {"value": fg, "score": (fg - 50) / 5, "label": fg_label}
            details.append(f"😱 Fear&Greed: {fg}/100 ({fg_label})")
    except:
        data["fear_greed"] = {"value": 50, "score": 0}

    try:
        vix = dl("^VIX")
        if len(vix) >= 5:
            v = float(vix["Close"].iloc[-1])
            sc = -5 if v > 30 else -3 if v > 25 else 3 if v < 15 else 1 if v < 18 else 0
            data["vix"] = {"value": v, "score": sc}
            details.append(f"{'✅' if sc >= 0 else '❌'} VIX: {round(v, 1)}")
    except:
        data["vix"] = {"score": 0}

    try:
        dxy = dl("DX-Y.NYB")
        if len(dxy) >= 20:
            prix = float(dxy['Close'].iloc[-1])
            ma20 = float(dxy['Close'].rolling(20).mean().iloc[-1])
            var_5j = float((dxy['Close'].iloc[-1] - dxy['Close'].iloc[-5]) / dxy['Close'].iloc[-5] * 100)
            sc = 0
            if prix < ma20:
                sc += 3
            if var_5j < -0.5:
                sc += 2
            elif var_5j > 0.5:
                sc -= 2
            if prix > ma20:
                sc -= 3
            data["dxy"] = {"prix": prix, "score": max(-10, min(10, sc))}
            details.append(f"{'✅' if sc > 0 else '❌'} Dollar: {round(prix, 1)} ({var_5j:+.1f}%)")
    except:
        data["dxy"] = {"score": 0}

    try:
        tnx = dl("^TNX")
        if len(tnx) >= 20:
            prix = float(tnx['Close'].iloc[-1])
            ma20 = float(tnx['Close'].rolling(20).mean().iloc[-1])
            sc = 3 if prix < ma20 else -3
            if prix > 4.5:
                sc -= 2
            elif prix < 3.5:
                sc += 2
            data["yields"] = {"prix": prix, "score": max(-10, min(10, sc))}
            details.append(f"{'✅' if sc > 0 else '❌'} Taux 10Y: {round(prix, 2)}%")
    except:
        data["yields"] = {"score": 0}

    try:
        spy = dl("SPY")
        if len(spy) >= 20:
            prix = float(spy['Close'].iloc[-1])
            ma20 = float(spy['Close'].rolling(20).mean().iloc[-1])
            var_5j = float((spy['Close'].iloc[-1] - spy['Close'].iloc[-5]) / spy['Close'].iloc[-5] * 100)
            sc = 3 if prix > ma20 else -3
            if var_5j > 2:
                sc += 2
            elif var_5j < -2:
                sc -= 2
            data["spy"] = {"prix": prix, "score": max(-10, min(10, sc))}
            details.append(f"{'✅' if sc > 0 else '❌'} S&P: {var_5j:+.1f}% (5j)")
    except:
        data["spy"] = {"score": 0}

    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        r = requests.get(url, params={"symbol": "BTCUSDT", "limit": 1}, timeout=5)
        if r.status_code == 200:
            rate = float(r.json()[0]["fundingRate"])
            sc = -3 if rate > 0.0005 else 3 if rate < -0.0001 else 0
            data["funding"] = {"current": rate * 100, "score": sc}
            details.append(f"📊 Funding BTC: {round(rate * 100, 4)}%")
    except:
        data["funding"] = {"score": 0}

    return data, details


def calc_macro_score(macro_data, categorie):
    if categorie == "crypto":
        weights = {'dxy': 0.15, 'yields': 0.10, 'vix': 0.10, 'fear_greed': 0.25, 'funding': 0.20, 'spy': 0.20}
    elif categorie == "commodities":
        weights = {'dxy': 0.30, 'yields': 0.25, 'vix': 0.15, 'spy': 0.15, 'fear_greed': 0.15}
    elif categorie == "forex":
        weights = {'dxy': 0.35, 'yields': 0.25, 'vix': 0.20, 'spy': 0.20}
    else:
        weights = {'spy': 0.30, 'vix': 0.25, 'yields': 0.20, 'dxy': 0.10, 'fear_greed': 0.15}
    total = 0; tw = 0
    for f, w in weights.items():
        if f in macro_data and 'score' in macro_data[f]:
            total += macro_data[f]['score'] * w; tw += w
    sensitivity = MACRO_SENSITIVITY.get(categorie, 0.7)
    return max(-10, min(10, (total / tw * sensitivity) if tw > 0 else 0))


@st.cache_data(ttl=600, show_spinner="📰 News...")
def get_news_score(ticker):
    try:
        kw = NEWS_KEYWORDS.get(ticker, [ticker])
        q = "+".join(kw[:3]).replace(" ", "+")
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en")
        if not feed.entries:
            return 0, "Pas d'articles", []
        vader = SentimentIntensityAnalyzer()
        scores = []; headlines = []
        for entry in feed.entries[:15]:
            title = entry.get("title", "")
            compound = vader.polarity_scores(title)["compound"]
            tl = title.lower()
            bull = sum(1 for k in BULLISH_KW if k in tl)
            bear = sum(1 for k in BEARISH_KW if k in tl)
            final = max(-1, min(1, compound + (bull - bear) * 0.25))
            scores.append(final)
            headlines.append({"title": title, "score": final})
        if scores:
            weights = np.linspace(1.5, 0.5, len(scores))
            avg = np.average(scores, weights=weights)
            score = max(-10, min(10, avg * 10))
            bull_n = sum(1 for s in scores if s > 0.1)
            bear_n = sum(1 for s in scores if s < -0.1)
            detail = f"{bull_n}+ / {len(scores) - bull_n - bear_n}= / {bear_n}-"
            return score, detail, sorted(headlines, key=lambda x: abs(x['score']), reverse=True)[:5]
    except:
        pass
    return 0, "Erreur", []


@st.cache_data(ttl=900)
def get_onchain(ticker):
    if ticker not in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        return 0, []
    score = 0; details = []
    if ticker == 'BTC-USD':
        try:
            r = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=5)
            if r.status_code == 200:
                f = r.json().get("fastestFee", 0)
                if f > 100:
                    score -= 2; details.append(f"🔴 Fees élevés ({f} sat/vB)")
                elif f < 10:
                    score += 1; details.append(f"🟢 Fees bas ({f} sat/vB)")
                else:
                    details.append(f"⚪ Fees {f} sat/vB")
        except:
            pass
        try:
            r = requests.get("https://mempool.space/api/mempool", timeout=5)
            if r.status_code == 200:
                count = r.json().get('count', 0)
                if count > 100000:
                    score -= 1; details.append(f"🔴 Mempool ({count} TX)")
                elif count < 10000:
                    score += 1; details.append(f"🟢 Mempool calme ({count} TX)")
        except:
            pass
        try:
            r = requests.get("https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json", timeout=10)
            if r.status_code == 200:
                values = [p['y'] for p in r.json().get('values', [])]
                if len(values) >= 14:
                    change = (np.mean(values[-7:]) - np.mean(values[:7])) / np.mean(values[:7]) * 100
                    if change > 5:
                        score += 2; details.append(f"🟢 Hashrate +{round(change, 1)}%")
                    elif change < -5:
                        score -= 2; details.append(f"🔴 Hashrate {round(change, 1)}%")
        except:
            pass
        try:
            r = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio", params={"symbol": "BTCUSDT", "period": "1h", "limit": 1}, timeout=5)
            if r.status_code == 200 and r.json():
                ls = float(r.json()[0]['longShortRatio'])
                if ls > 2.0:
                    score -= 2; details.append(f"🔴 L/S {round(ls, 2)} (trop longs)")
                elif ls < 0.8:
                    score += 2; details.append(f"🟢 L/S {round(ls, 2)} (shorts)")
                else:
                    details.append(f"⚪ L/S {round(ls, 2)}")
        except:
            pass
    try:
        r = requests.get("https://api.llama.fi/v2/historicalChainTvl", timeout=10)
        if r.status_code == 200:
            tvl_data = r.json()
            if len(tvl_data) >= 7:
                recent = tvl_data[-1].get('tvl', 0); week_ago = tvl_data[-7].get('tvl', 0)
                change = (recent - week_ago) / week_ago * 100 if week_ago > 0 else 0
                if change > 5:
                    score += 2; details.append(f"🟢 TVL +{round(change, 1)}%")
                elif change < -5:
                    score -= 2; details.append(f"🔴 TVL {round(change, 1)}%")
    except:
        pass
    return max(-10, min(10, score)), details


# ══════════════════════════════════════════════════════════
# MACHINE LEARNING
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="🤖 ML...")
def ml_predict(ticker, data):
    try:
        if len(data) < 120:
            return None
        df = data.copy()
        df["r1"] = df["Close"].pct_change(1)
        df["r5"] = df["Close"].pct_change(5)
        df["r10"] = df["Close"].pct_change(10)
        df["vol10"] = df["r1"].rolling(10).std()
        df["mom10"] = df["Close"] / df["Close"].shift(10) - 1
        df["pma20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
        df["pma50"] = df["Close"] / df["Close"].rolling(50).mean() - 1
        df["vratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["rsi_d"] = df["RSI"].diff(3)
        df["macd_h"] = df["MACD"] - df["MACD_Signal"]
        df["macd_hd"] = df["macd_h"].diff(3)
        bb_r = df["BB_Upper"] - df["BB_Lower"]
        df["bb_pos"] = (df["Close"] - df["BB_Lower"]) / bb_r
        df["atr_p"] = df["ATR"] / df["Close"] * 100
        df["target"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.01).astype(int)
        feats = ["RSI", "Stoch_K", "ADX", "r1", "r5", "r10", "vol10", "mom10", "pma20", "pma50", "vratio", "rsi_d", "macd_h", "macd_hd", "bb_pos", "atr_p"]
        clean = df[feats + ["target"]].dropna()
        if len(clean) < 60:
            return None
        X = clean[feats]; y = clean["target"]
        sp = int(len(X) * 0.8)
        Xtr, Xte = X.iloc[:sp], X.iloc[sp:]
        ytr, yte = y.iloc[:sp], y.iloc[sp:]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        if HAS_LGBM:
            model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1)
        else:
            model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8)
        model.fit(Xtr_s, ytr)
        acc = accuracy_score(yte, model.predict(Xte_s))
        proba = model.predict_proba(scaler.transform(X.iloc[[-1]]))[0]
        top_f = sorted(zip(feats, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
        return {"hausse": proba[1], "baisse": proba[0], "acc": acc, "dir": "ACHAT" if proba[1] > 0.55 else "VENTE" if proba[0] > 0.55 else "NEUTRE", "conf": max(proba), "top_features": top_f}
    except:
        return None


# ══════════════════════════════════════════════════════════
# MULTI-TIMEFRAME
# ══════════════════════════════════════════════════════════

def analyser_mtf(ticker):
    result = {'4h': None, 'weekly': None, 'consensus': "NEUTRE"}
    try:
        data_4h = telecharger_4h(ticker)
        if not data_4h.empty and len(data_4h) >= 30:
            data_4h['RSI'] = calc_rsi(data_4h['Close'])
            data_4h['MACD'], data_4h['MACD_Signal'] = calc_macd(data_4h['Close'])
            d = data_4h.iloc[-1]
            rsi_4h = float(d['RSI']) if not np.isnan(float(d['RSI'])) else 50
            macd_4h = float(d['MACD']); sig_4h = float(d['MACD_Signal'])
            sa = 0; sv = 0
            if rsi_4h < 35: sa += 1
            elif rsi_4h > 65: sv += 1
            if macd_4h > sig_4h: sa += 1
            else: sv += 1
            t = "ACHAT" if sa >= 2 else "VENTE" if sv >= 2 else "NEUTRE"
            result['4h'] = {'tendance': t, 'rsi': rsi_4h}
    except:
        pass
    try:
        data_w = telecharger_weekly(ticker)
        if not data_w.empty and len(data_w) >= 20:
            data_w['RSI'] = calc_rsi(data_w['Close'])
            data_w['MA_20'] = data_w['Close'].rolling(20).mean()
            d = data_w.iloc[-1]
            rsi_w = float(d['RSI']) if not np.isnan(float(d['RSI'])) else 50
            px = float(d['Close']); ma20 = float(d['MA_20']) if not np.isnan(float(d['MA_20'])) else px
            if rsi_w < 40 and px > ma20: t = "ACHAT"
            elif rsi_w > 60 and px < ma20: t = "VENTE"
            elif px > ma20: t = "ACHAT"
            elif px < ma20: t = "VENTE"
            else: t = "NEUTRE"
            result['weekly'] = {'tendance': t, 'rsi': rsi_w}
    except:
        pass
    tendances = []
    if result['4h']:
        tendances.append(result['4h']['tendance'])
    if result['weekly']:
        tendances.append(result['weekly']['tendance'])
    if tendances.count("ACHAT") >= 2:
        result['consensus'] = "ACHAT"
    elif tendances.count("VENTE") >= 2:
        result['consensus'] = "VENTE"
    elif "ACHAT" in tendances and "VENTE" not in tendances:
        result['consensus'] = "ACHAT"
    elif "VENTE" in tendances and "ACHAT" not in tendances:
        result['consensus'] = "VENTE"
    return result
# ══════════════════════════════════════════════════════════
# ÉVALUATION COMPLÈTE
# ══════════════════════════════════════════════════════════

def V(v):
    return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)


def evaluer(data, ticker, nom, macro_data, divergences=None):
    if len(data) < 2:
        return None
    last = data.iloc[-1]; prev = data.iloc[-2]
    prix = V(last["Close"]); adx = V(last["ADX"])
    atr = V(last["ATR"]) if not np.isnan(V(last["ATR"])) else 0
    if np.isnan(adx) or adx < SEUIL_ADX:
        return {"action": "PLAT", "prix": prix, "adx": adx, "atr": atr, "score_achat": 0, "score_vente": 0, "details": [("ADX", "PLAT", f"ADX={round(adx, 1)} < {SEUIL_ADX}")], "sl_tp": None, "ml": None, "supports": [], "resistances": [], "divergences_txt": "", "mtf": None}

    sa = 0; sv = 0; det = []

    # RSI
    rsi = V(last["RSI"]); rsi_p = V(prev["RSI"])
    if rsi < 30 and rsi_p < 30:
        sa += POIDS["RSI"]; det.append(("RSI", "ACHAT", f"Survendu confirmé ({round(rsi, 1)})"))
    elif rsi > 70 and rsi_p > 70:
        sv += POIDS["RSI"]; det.append(("RSI", "VENTE", f"Suracheté confirmé ({round(rsi, 1)})"))
    else:
        det.append(("RSI", "—", f"{round(rsi, 1)}"))

    # MACD
    m = V(last["MACD"]); ms = V(last["MACD_Signal"])
    mp = V(prev["MACD"]); msp = V(prev["MACD_Signal"])
    if m > ms and mp > msp:
        sa += POIDS["MACD"]; det.append(("MACD", "ACHAT", "Bullish confirmé"))
    elif m < ms and mp < msp:
        sv += POIDS["MACD"]; det.append(("MACD", "VENTE", "Bearish confirmé"))
    else:
        det.append(("MACD", "—", "Croisement récent"))

    # Stochastique
    sk = V(last["Stoch_K"]); skp = V(prev["Stoch_K"])
    if sk < 20 and skp < 20:
        sa += POIDS["STOCH"]; det.append(("STOCH", "ACHAT", f"Survendu ({round(sk, 1)})"))
    elif sk > 80 and skp > 80:
        sv += POIDS["STOCH"]; det.append(("STOCH", "VENTE", f"Suracheté ({round(sk, 1)})"))
    else:
        det.append(("STOCH", "—", f"{round(sk, 1)}"))

    # Fibonacci
    f6 = V(last["Fib_618"]); f3 = V(last["Fib_382"])
    if prix <= f6:
        sa += POIDS["FIBO"]; det.append(("FIBO", "ACHAT", "Sous 61.8%"))
    elif prix >= f3:
        sv += POIDS["FIBO"]; det.append(("FIBO", "VENTE", "Au-dessus 38.2%"))
    else:
        det.append(("FIBO", "—", "Entre niveaux"))

    # MA200
    ma = V(last["MA_200"])
    if not np.isnan(ma):
        if prix <= ma * 1.02:
            sa += POIDS["MA200"]; det.append(("MA200", "ACHAT", "Sous/proche MA200"))
        elif prix >= ma * 1.10:
            sv += POIDS["MA200"]; det.append(("MA200", "VENTE", "+10% au-dessus"))
        else:
            det.append(("MA200", "—", "Zone normale"))

    # Bollinger
    bbu = V(last["BB_Upper"]); bbl = V(last["BB_Lower"])
    if not np.isnan(bbu):
        if prix <= bbl:
            sa += POIDS["BOLLINGER"]; det.append(("BOLL", "ACHAT", "Bande basse"))
        elif prix >= bbu:
            sv += POIDS["BOLLINGER"]; det.append(("BOLL", "VENTE", "Bande haute"))
        else:
            det.append(("BOLL", "—", "Entre bandes"))

    # Volume
    vol = V(last["Volume"]); vm = V(last["Vol_Moy"])
    if not np.isnan(vm) and vm > 0:
        ratio_vol = vol / vm
        if ratio_vol >= 1.5:
            if sa > sv:
                sa += POIDS["VOLUME"]; det.append(("VOL", "ACHAT", f"{round(ratio_vol, 1)}x confirme"))
            elif sv > sa:
                sv += POIDS["VOLUME"]; det.append(("VOL", "VENTE", f"{round(ratio_vol, 1)}x confirme"))
        else:
            det.append(("VOL", "—", f"{round(ratio_vol, 1)}x"))

    # Divergences techniques RSI/MACD
    div_tech = detecter_divergences_tech(data)
    if div_tech['rsi'] == "HAUSSIERE" or div_tech['macd'] == "HAUSSIERE":
        sa += POIDS["DIVERGENCE"]; det.append(("DIV_TECH", "ACHAT", "Divergence haussière"))
    elif div_tech['rsi'] == "BAISSIERE" or div_tech['macd'] == "BAISSIERE":
        sv += POIDS["DIVERGENCE"]; det.append(("DIV_TECH", "VENTE", "Divergence baissière"))
    else:
        det.append(("DIV_TECH", "—", "Pas de divergence"))

    # Ichimoku
    try:
        tk = V(last["Ichi_TK"]); kj = V(last["Ichi_KJ"])
        spa = V(last["Ichi_SpA"]); spb = V(last["Ichi_SpB"])
        if not np.isnan(spa) and not np.isnan(spb):
            nh = max(spa, spb); nb = min(spa, spb)
            if prix > nh and tk > kj:
                sa += POIDS["ICHIMOKU"]; det.append(("ICHI", "ACHAT", "Au-dessus nuage + TK>KJ"))
            elif prix < nb and tk < kj:
                sv += POIDS["ICHIMOKU"]; det.append(("ICHI", "VENTE", "Sous nuage + TK<KJ"))
            elif prix > nh:
                sa += POIDS["ICHIMOKU"] * 0.5; det.append(("ICHI", "ACHAT", "Au-dessus nuage"))
            elif prix < nb:
                sv += POIDS["ICHIMOKU"] * 0.5; det.append(("ICHI", "VENTE", "Sous nuage"))
            else:
                det.append(("ICHI", "—", "Dans le nuage"))
    except:
        det.append(("ICHI", "—", "N/A"))

    # VWAP
    vwap = V(last["VWAP"])
    if not np.isnan(vwap):
        if prix < vwap * 0.99:
            sa += POIDS["VWAP"]; det.append(("VWAP", "ACHAT", "Sous VWAP"))
        elif prix > vwap * 1.01:
            sv += POIDS["VWAP"]; det.append(("VWAP", "VENTE", "Au-dessus VWAP"))
        else:
            det.append(("VWAP", "—", "Proche VWAP"))

    # Supports / Résistances
    supports, resistances = detect_supports_resistances(data)
    sp_close = max([s for s in supports if s < prix], default=None)
    rs_close = min([r for r in resistances if r > prix], default=None)
    if sp_close and (prix - sp_close) / prix < 0.02:
        sa += POIDS["SUPPORTS_RES"]; det.append(("S/R", "ACHAT", f"Support {round(sp_close, 2)}"))
    elif rs_close and (rs_close - prix) / prix < 0.02:
        sv += POIDS["SUPPORTS_RES"]; det.append(("S/R", "VENTE", f"Résistance {round(rs_close, 2)}"))
    else:
        det.append(("S/R", "—", f"S:{round(sp_close, 2) if sp_close else '?'} R:{round(rs_close, 2) if rs_close else '?'}"))

    # Order Flow (crypto)
    if ticker in CCXT_SYMBOLS:
        ob = fetch_orderbook(CCXT_SYMBOLS[ticker])
        if ob:
            imb = ob["imbalance"]
            if imb > 0.3:
                sa += POIDS["ORDER_FLOW"]; det.append(("FLOW", "ACHAT", f"+{round(imb * 100, 0)}%"))
            elif imb < -0.3:
                sv += POIDS["ORDER_FLOW"]; det.append(("FLOW", "VENTE", f"{round(imb * 100, 0)}%"))
            else:
                det.append(("FLOW", "—", f"{round(imb * 100, 0)}%"))
    else:
        det.append(("FLOW", "—", "N/A"))

    # Macro
    categorie = ACTIF_CATEGORIE.get(nom, "forex")
    ms_val = calc_macro_score(macro_data, categorie)
    if ms_val >= 3:
        sa += POIDS["MACRO"]; det.append(("MACRO", "ACHAT", f"+{round(ms_val, 1)}/10"))
    elif ms_val <= -3:
        sv += POIDS["MACRO"]; det.append(("MACRO", "VENTE", f"{round(ms_val, 1)}/10"))
    elif ms_val >= 1:
        sa += POIDS["MACRO"] * 0.4; det.append(("MACRO", "ACHAT", f"Léger +{round(ms_val, 1)}"))
    elif ms_val <= -1:
        sv += POIDS["MACRO"] * 0.4; det.append(("MACRO", "VENTE", f"Léger {round(ms_val, 1)}"))
    else:
        det.append(("MACRO", "—", f"{round(ms_val, 1)}"))

    # Sentiment
    if macro_data.get("fear_greed"):
        fg = macro_data["fear_greed"].get("value", 50)
        if fg < 25:
            sa += POIDS["SENTIMENT"]; det.append(("SENT", "ACHAT", f"Extreme Fear ({fg})"))
        elif fg < 35:
            sa += POIDS["SENTIMENT"] * 0.5; det.append(("SENT", "ACHAT", f"Fear ({fg})"))
        elif fg > 75:
            sv += POIDS["SENTIMENT"]; det.append(("SENT", "VENTE", f"Extreme Greed ({fg})"))
        elif fg > 65:
            sv += POIDS["SENTIMENT"] * 0.5; det.append(("SENT", "VENTE", f"Greed ({fg})"))
        else:
            det.append(("SENT", "—", f"F&G={fg}"))

    # News NLP
    ns, ns_detail, ns_headlines = get_news_score(ticker)
    if ns >= 4:
        sa += POIDS["NEWS_NLP"]; det.append(("NEWS", "ACHAT", f"+{round(ns, 1)} ({ns_detail})"))
    elif ns <= -4:
        sv += POIDS["NEWS_NLP"]; det.append(("NEWS", "VENTE", f"{round(ns, 1)} ({ns_detail})"))
    elif ns >= 2:
        sa += POIDS["NEWS_NLP"] * 0.4; det.append(("NEWS", "ACHAT", f"Léger + ({round(ns, 1)})"))
    elif ns <= -2:
        sv += POIDS["NEWS_NLP"] * 0.4; det.append(("NEWS", "VENTE", f"Léger - ({round(ns, 1)})"))
    else:
        det.append(("NEWS", "—", f"{round(ns, 1)}"))

    # On-Chain
    oc_score, oc_details = get_onchain(ticker)
    if oc_score >= 3:
        sa += POIDS["ONCHAIN"]; det.append(("CHAIN", "ACHAT", f"+{oc_score}"))
    elif oc_score <= -3:
        sv += POIDS["ONCHAIN"]; det.append(("CHAIN", "VENTE", f"{oc_score}"))
    elif oc_score >= 1:
        sa += POIDS["ONCHAIN"] * 0.4; det.append(("CHAIN", "ACHAT", f"Léger +"))
    elif oc_score <= -1:
        sv += POIDS["ONCHAIN"] * 0.4; det.append(("CHAIN", "VENTE", f"Léger -"))
    else:
        det.append(("CHAIN", "—", "N/A"))

    # ML
    ml = ml_predict(ticker, data)
    if ml and ml["acc"] > 0.52:
        if ml["dir"] == "ACHAT" and ml["conf"] > 0.55:
            sa += POIDS["ML_PREDICTION"]; det.append(("🤖 ML", "ACHAT", f"{round(ml['hausse'] * 100, 0)}% (acc {round(ml['acc'] * 100, 0)}%)"))
        elif ml["dir"] == "VENTE" and ml["conf"] > 0.55:
            sv += POIDS["ML_PREDICTION"]; det.append(("🤖 ML", "VENTE", f"{round(ml['baisse'] * 100, 0)}% (acc {round(ml['acc'] * 100, 0)}%)"))
        else:
            det.append(("🤖 ML", "—", f"Confiance {round(ml['conf'] * 100, 0)}%"))
    else:
        det.append(("🤖 ML", "—", "Données insuffisantes"))

    # Or/BTC
    sig_ob, msg_ob = indicateur_or_btc(ticker)
    if sig_ob == 1:
        sa += POIDS["OR_BTC"]; det.append(("OR/BTC", "ACHAT", msg_ob))
    elif sig_ob == -1:
        sv += POIDS["OR_BTC"]; det.append(("OR/BTC", "VENTE", msg_ob))
    else:
        det.append(("OR/BTC", "—", "Neutre"))

    # Divergences Métaux (DXY/Or/Argent)
    div_txt = ""
    if divergences and ticker in ["GC=F", "SI=F", "PL=F"]:
        div_signal, div_detail = get_divergence_metal_impact(ticker, divergences)
        if div_signal == 1:
            sa += POIDS["DIV_METALS"]; det.append(("DIV_METALS", "ACHAT", div_detail)); div_txt = f"🟢 {div_detail}"
        elif div_signal == -1:
            sv += POIDS["DIV_METALS"]; det.append(("DIV_METALS", "VENTE", div_detail)); div_txt = f"🔴 {div_detail}"
        else:
            det.append(("DIV_METALS", "—", "Normal"))

    # SL / TP
    sl_tp = None
    if sa > sv:
        sl = prix - 1.5 * atr; tp = prix + 2.5 * atr
        if sp_close:
            sl = max(sl, sp_close * 0.998)
        if rs_close:
            tp = min(tp, rs_close * 0.998)
        risque = (prix - sl) / prix * 100; reward = (tp - prix) / prix * 100
        sl_tp = {"stop_loss": sl, "take_profit": tp, "risque_pct": risque, "reward_pct": reward, "ratio_rr": reward / risque if risque > 0 else 0, "atr": atr}
    elif sv > sa:
        sl = prix + 1.5 * atr; tp = prix - 2.5 * atr
        if rs_close:
            sl = min(sl, rs_close * 1.002)
        if sp_close:
            tp = max(tp, sp_close * 1.002)
        risque = (sl - prix) / prix * 100; reward = (prix - tp) / prix * 100
        sl_tp = {"stop_loss": sl, "take_profit": tp, "risque_pct": risque, "reward_pct": reward, "ratio_rr": reward / risque if risque > 0 else 0, "atr": atr}

    # MTF
    mtf = analyser_mtf(ticker)

    # Action
    seuil = config.get("seuil", 8.0)
    if sa >= seuil and sa > sv:
        action = "ACHAT"
    elif sv >= seuil:
        action = "VENTE"
    else:
        action = "ATTENDRE"

    det.append(("ADX", "OK", f"ADX={round(adx, 1)} → Tendance"))
    if mtf:
        if mtf.get('4h'):
            det.append(("MTF 4H", mtf['4h']['tendance'], f"RSI: {round(mtf['4h']['rsi'], 0)}"))
        if mtf.get('weekly'):
            det.append(("MTF W", mtf['weekly']['tendance'], f"RSI: {round(mtf['weekly']['rsi'], 0)}"))

    return {"action": action, "prix": prix, "adx": adx, "atr": atr, "score_achat": sa, "score_vente": sv, "details": det, "sl_tp": sl_tp, "ml": ml, "supports": supports, "resistances": resistances, "divergences_txt": div_txt, "mtf": mtf}


# ══════════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════════

def scan_actif(nom, ticker, macro_data, divergences):
    try:
        data = telecharger(ticker)
        if data.empty:
            return None
        data = calculer_indicateurs(data)
        if np.isnan(V(data.iloc[-1]["RSI"])):
            return None
        result = evaluer(data, ticker, nom, macro_data, divergences)
        if not result:
            return None
        tz = pytz.timezone("Europe/Zurich"); heure = datetime.now(tz).hour
        cat = ACTIF_CATEGORIE.get(nom, "forex")
        h_info = HORAIRES.get(cat, HORAIRES["forex"])
        result.update({"nom": nom, "ticker": ticker, "score_max": SCORE_MAX, "data": data, "heure": heure, "heure_ok_buy": heure in h_info["buy"], "heure_ok_sell": heure in h_info["sell"], "heure_avoid": heure in h_info["avoid"], "buy_txt": h_info["buy_txt"], "sell_txt": h_info["sell_txt"]})
        return result
    except:
        return None


def lancer_scan(actifs, macro_data, divergences):
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(scan_actif, nom, ACTIFS[nom], macro_data, divergences): nom for nom in actifs}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    results.sort(key=lambda x: (0 if x["action"] in ["ACHAT", "VENTE"] else 1, -max(x["score_achat"], x["score_vente"])))
    return results


# ══════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="📊 Backtesting...")
def backtester(ticker, seuil_score):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty or len(data) < 100:
            return None
        data = calculer_indicateurs(data)
        trades = []; position = None; prix_entree = 0
        for i in range(50, len(data) - 1):
            d = data.iloc[i]; p = data.iloc[i - 1]
            prix = float(d['Close']); adx_val = float(d['ADX']) if not np.isnan(float(d['ADX'])) else 0
            atr_val = float(d['ATR']) if not np.isnan(float(d['ATR'])) else 0
            if adx_val < SEUIL_ADX:
                if position == "LONG":
                    trades.append({'type': 'LONG', 'pnl': ((prix - prix_entree) / prix_entree) * 100}); position = None
                elif position == "SHORT":
                    trades.append({'type': 'SHORT', 'pnl': ((prix_entree - prix) / prix_entree) * 100}); position = None
                continue
            sa = 0; sv = 0
            rsi = float(d['RSI']) if not np.isnan(float(d['RSI'])) else 50
            rsi_p = float(p['RSI']) if not np.isnan(float(p['RSI'])) else 50
            if rsi < 30 and rsi_p < 30: sa += POIDS["RSI"]
            elif rsi > 70 and rsi_p > 70: sv += POIDS["RSI"]
            macd = float(d['MACD']) if not np.isnan(float(d['MACD'])) else 0
            sig = float(d['MACD_Signal']) if not np.isnan(float(d['MACD_Signal'])) else 0
            macd_p = float(p['MACD']) if not np.isnan(float(p['MACD'])) else 0
            sig_p = float(p['MACD_Signal']) if not np.isnan(float(p['MACD_Signal'])) else 0
            if macd > sig and macd_p > sig_p: sa += POIDS["MACD"]
            elif macd < sig and macd_p < sig_p: sv += POIDS["MACD"]
            sk = float(d['Stoch_K']) if not np.isnan(float(d['Stoch_K'])) else 50
            skp = float(p['Stoch_K']) if not np.isnan(float(p['Stoch_K'])) else 50
            if sk < 20 and skp < 20: sa += POIDS["STOCH"]
            elif sk > 80 and skp > 80: sv += POIDS["STOCH"]
            f6 = float(d['Fib_618']) if not np.isnan(float(d['Fib_618'])) else prix
            f3 = float(d['Fib_382']) if not np.isnan(float(d['Fib_382'])) else prix
            if prix <= f6: sa += POIDS["FIBO"]
            elif prix >= f3: sv += POIDS["FIBO"]
            ma200 = float(d['MA_200']) if not np.isnan(float(d['MA_200'])) else prix
            if prix <= ma200 * 1.02: sa += POIDS["MA200"]
            elif prix >= ma200 * 1.10: sv += POIDS["MA200"]
            bbl = float(d['BB_Lower']) if not np.isnan(float(d['BB_Lower'])) else prix
            bbu = float(d['BB_Upper']) if not np.isnan(float(d['BB_Upper'])) else prix
            if prix <= bbl: sa += POIDS["BOLLINGER"]
            elif prix >= bbu: sv += POIDS["BOLLINGER"]
            try:
                tk = float(d['Ichi_TK']); kj = float(d['Ichi_KJ'])
                spa_v = float(d['Ichi_SpA']); spb_v = float(d['Ichi_SpB'])
                if not np.isnan(spa_v) and not np.isnan(spb_v):
                    if prix > max(spa_v, spb_v) and tk > kj: sa += POIDS["ICHIMOKU"]
                    elif prix < min(spa_v, spb_v) and tk < kj: sv += POIDS["ICHIMOKU"]
            except:
                pass
            if position is None:
                if sa >= seuil_score and sa > sv: position = "LONG"; prix_entree = prix
                elif sv >= seuil_score: position = "SHORT"; prix_entree = prix
            elif position == "LONG":
                sl = prix_entree - 1.5 * atr_val if atr_val > 0 else prix_entree * 0.97
                tp = prix_entree + 2.5 * atr_val if atr_val > 0 else prix_entree * 1.05
                if prix <= sl: trades.append({'type': 'LONG', 'pnl': ((sl - prix_entree) / prix_entree) * 100}); position = None
                elif prix >= tp: trades.append({'type': 'LONG', 'pnl': ((tp - prix_entree) / prix_entree) * 100}); position = None
                elif sv >= seuil_score: trades.append({'type': 'LONG', 'pnl': ((prix - prix_entree) / prix_entree) * 100}); position = None
            elif position == "SHORT":
                sl = prix_entree + 1.5 * atr_val if atr_val > 0 else prix_entree * 1.03
                tp = prix_entree - 2.5 * atr_val if atr_val > 0 else prix_entree * 0.95
                if prix >= sl: trades.append({'type': 'SHORT', 'pnl': ((prix_entree - sl) / prix_entree) * 100}); position = None
                elif prix <= tp: trades.append({'type': 'SHORT', 'pnl': ((prix_entree - tp) / prix_entree) * 100}); position = None
                elif sa >= seuil_score: trades.append({'type': 'SHORT', 'pnl': ((prix_entree - prix) / prix_entree) * 100}); position = None
        if position:
            pf = float(data.iloc[-1]['Close'])
            if position == "LONG": trades.append({'type': 'LONG', 'pnl': ((pf - prix_entree) / prix_entree) * 100})
            else: trades.append({'type': 'SHORT', 'pnl': ((prix_entree - pf) / prix_entree) * 100})
        if not trades:
            return None
        df_t = pd.DataFrame(trades)
        nb = len(df_t); gagnants = len(df_t[df_t['pnl'] > 0]); perdants = nb - gagnants
        wr = gagnants / nb * 100; pnl_tot = df_t['pnl'].sum(); pnl_moy = df_t['pnl'].mean()
        gains = df_t[df_t['pnl'] > 0]['pnl'].sum(); pertes = abs(df_t[df_t['pnl'] <= 0]['pnl'].sum())
        pf = gains / pertes if pertes > 0 else float('inf')
        cumul = df_t['pnl'].cumsum(); dd = (cumul - cumul.cummax()).min()
        sharpe = (pnl_moy / df_t['pnl'].std()) * np.sqrt(nb) if df_t['pnl'].std() > 0 else 0
        return {'nb': nb, 'gagnants': gagnants, 'perdants': perdants, 'wr': wr, 'pnl': pnl_tot, 'pnl_moy': pnl_moy, 'best': df_t['pnl'].max(), 'worst': df_t['pnl'].min(), 'pf': pf, 'dd': dd, 'sharpe': sharpe, 'trades': df_t}
    except:
        return None


# ══════════════════════════════════════════════════════════
# GRAPHIQUE
# ══════════════════════════════════════════════════════════

def graphique(data, nom, supports=None, resistances=None):
    df = data.tail(60)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.4, 0.2, 0.2, 0.2], subplot_titles=["Prix + Indicateurs", "RSI", "MACD", "Volume"])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Prix", line=dict(color="#667eea", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_200"], name="MA200", line=dict(color="#ffd700", dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB+", line=dict(color="rgba(255,100,100,0.5)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB-", line=dict(color="rgba(100,255,100,0.5)", width=1), fill="tonexty", fillcolor="rgba(100,100,255,0.05)"), row=1, col=1)
    if "Ichi_SpA" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Ichi_SpA"], name="SpanA", line=dict(color="rgba(0,255,100,0.3)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Ichi_SpB"], name="SpanB", line=dict(color="rgba(255,100,0,0.3)", width=1), fill="tonexty", fillcolor="rgba(100,200,100,0.1)"), row=1, col=1)
    if supports:
        for s in supports[:3]:
            fig.add_hline(y=s, line_dash="dot", line_color="cyan", annotation_text=f"S:{round(s, 2)}", row=1, col=1)
    if resistances:
        for r in resistances[:3]:
            fig.add_hline(y=r, line_dash="dot", line_color="red", annotation_text=f"R:{round(r, 2)}", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#a855f7", width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="cyan", row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#667eea", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#f5576c", width=1.5)), row=3, col=1)
    hist = df["MACD"] - df["MACD_Signal"]
    colors_h = ['rgba(0,210,255,0.6)' if float(h) >= 0 else 'rgba(245,87,108,0.6)' for h in hist]
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist", marker_color=colors_h), row=3, col=1)
    vol_c = ['rgba(0,210,255,0.6)' if float(df['Close'].iloc[i]) >= float(df['Open'].iloc[i]) else 'rgba(245,87,108,0.6)' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol", marker_color=vol_c), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_Moy"], name="VolMoy", line=dict(color="#ffd700", width=1.5, dash="dash")), row=4, col=1)
    fig.update_layout(height=800, showlegend=True, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=50, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


# ══════════════════════════════════════════════════════════
# CHECK-LIST
# ══════════════════════════════════════════════════════════

def calculer_checklist(r, risque_pct):
    score = max(r.get("score_achat", 0), r.get("score_vente", 0))
    adx = r.get("adx", 0)
    rr = r.get("sl_tp", {}).get("ratio_rr", 0) if r.get("sl_tp") else 0
    ml = r.get("ml")
    action = r.get("action", "ATTENDRE")
    heure_ok = (r.get("heure_ok_buy") and action == "ACHAT") or (r.get("heure_ok_sell") and action == "VENTE")
    heure_avoid = r.get("heure_avoid", False)
    div_ok = True
    if r.get("ticker") in ["GC=F", "SI=F", "PL=F"]:
        for ind, sig, txt in r.get("details", []):
            if ind == "DIV_METALS" and sig == "VENTE" and action == "ACHAT":
                div_ok = False
            elif ind == "DIV_METALS" and sig == "ACHAT" and action == "VENTE":
                div_ok = False
    ml_ok = bool(ml and ml.get("acc", 0) > 0.52 and ml.get("dir") == action)
    risque_ok = r["sl_tp"].get("risque_pct", 0) <= risque_pct * 1.5 if r.get("sl_tp") else True
    mtf_ok = r.get("mtf", {}).get("consensus") == action if r.get("mtf") else False
    return [
        ("📊 Score ≥ 12", score >= 12),
        ("⚖️ R:R ≥ 1:1.5", rr >= 1.5),
        ("📈 ADX > 25", adx > 25),
        ("⏰ Bonne heure", heure_ok and not heure_avoid),
        ("🤖 ML confirme", ml_ok),
        ("🔀 Divergences OK", div_ok),
        ("📊 MTF aligné", mtf_ok),
        ("💰 Risque OK", risque_ok),
    ]


def afficher_checklist(r, risque_pct):
    criteres = calculer_checklist(r, risque_pct)
    score_ck = sum(1 for _, ok in criteres if ok)
    action = r.get("action", "?")
    emoji = "🟢" if action == "ACHAT" else "🔴" if action == "VENTE" else "⏸️"
    st.markdown(f"### {emoji} {r.get('nom', '?')} — {action}")
    col_a, col_b = st.columns(2)
    for i, (label, ok) in enumerate(criteres):
        target = col_a if i < 4 else col_b
        with target:
            st.markdown(f"{'✅' if ok else '❌'} {label}")
    st.markdown("---")
    if score_ck >= 6:
        st.success(f"🟢 **GO** — {score_ck}/8 critères validés")
    elif score_ck >= 4:
        st.warning(f"🟡 **PRUDENCE** — {score_ck}/8")
    else:
        st.error(f"🔴 **STOP** — {score_ck}/8 (attends)")
    st.progress(score_ck / 8)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Score", f"{round(max(r.get('score_achat', 0), r.get('score_vente', 0)), 1)}/{round(SCORE_MAX, 0)}")
    mc2.metric("ADX", f"{round(r.get('adx', 0), 1)}")
    mc3.metric("R:R", f"1:{round(r.get('sl_tp', {}).get('ratio_rr', 0) if r.get('sl_tp') else 0, 1)}")
    return score_ck


def calculer_taille_position(capital, risque_pct, prix, stop_loss):
    risque_par_unite = abs(prix - stop_loss)
    if risque_par_unite == 0:
        return 0, 0
    montant_risque = capital * (risque_pct / 100)
    nb = montant_risque / risque_par_unite
    return nb, nb * prix
# ══════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ══════════════════════════════════════════════════════════

st.title("🧠 Trading Scanner v7.0")
st.caption(f"{'✅MT5' if HAS_MT5 else '⚠️MT5'} | {'✅ccxt' if HAS_CCXT else '⚠️ccxt'} | {'✅LGB' if HAS_LGBM else '⚠️GB'} | {'✅pandas-ta' if HAS_PANDAS_TA else '⚠️'} | Score max: {SCORE_MAX} pts | {len(POIDS)} indicateurs")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    seuil = st.slider("Seuil d'alerte", 4.0, 20.0, config.get("seuil", 8.0), 0.5)
    config["seuil"] = seuil; save_config()
    st.caption(f"Score max: {SCORE_MAX} pts")

    st.divider()
    st.header("💰 Capital & Risque")
    capital = st.number_input("Capital (CHF)", 0, 1000000, config.get("capital", 1000), 100)
    config["capital"] = capital; save_config()
    risque_pct = st.slider("Risque/trade %", 0.5, 5.0, 2.0, 0.5)

    st.divider()
    st.header("📊 Actifs")
    defaut = config.get("actifs", ["🥇 Or (Gold)", "₿ Bitcoin", "💵 EUR/USD"])
    defaut = [a for a in defaut if a in ACTIFS] or ["🥇 Or (Gold)", "₿ Bitcoin", "💵 EUR/USD"]
    actifs_choisis = st.multiselect("Sélection", list(ACTIFS.keys()), default=defaut)
    config["actifs"] = actifs_choisis; save_config()

    st.divider()
    st.header("🔔 Alertes")
    alert_mode = st.radio("Mode", ["Aucune", "Email Bluewin", "Telegram"])
    email_addr = ""; email_pass = ""; tg_token = ""; tg_chat = ""
    if alert_mode == "Email Bluewin":
        email_addr = st.text_input("Email", placeholder="nom@bluewin.ch")
        email_pass = st.text_input("Mot de passe", type="password")
    elif alert_mode == "Telegram":
        tg_token = st.text_input("Bot Token", value=config.get("tg_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=config.get("tg_chat", ""))
        config["tg_token"] = tg_token; config["tg_chat"] = tg_chat; save_config()

    st.divider()
    st.header("🌍 Macro Live")
    macro_data, macro_details = fetch_macro()
    for d in macro_details[:6]:
        st.caption(d)

    st.divider()
    st.header("⏰ Auto-Scan")
    auto_scan = st.toggle("Activer", False)
    auto_freq = st.selectbox("Fréquence", ["30 sec", "1 min", "5 min", "15 min"], index=2)
    auto_sec = {"30 sec": 30, "1 min": 60, "5 min": 300, "15 min": 900}[auto_freq]

    st.divider()
    with st.expander("📖 AIDE"):
        st.markdown("""
**Indicateurs (rappel) :**
- RSI < 30 = achat | > 70 = vente
- Stoch < 20 = achat | > 80 = vente
- MACD > Signal = achat | < Signal = vente
- Fibo < 61.8% = achat | > 38.2% = vente
- MA200 : sous = achat | +10% dessus = vente
- ADX > 25 = tendance confirmée

**Nouveautés v7.0 :**
- 🔌 MT5 (trading réel/démo)
- ✅ Check-list 8 critères
- 📊 Backtesting intégré
- 🔀 Divergences DXY/Or/Argent
- 🤖 ML + Ichimoku + Order Flow
- 📱 Telegram + Email Bluewin
- 🎮 Simulation avec levier + frais
        """)

# --- BOUTONS ---
c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
with c1:
    btn_scan = st.button("🚀 SCANNER", type="primary", use_container_width=True)
with c2:
    btn_backtest = st.button("📊 Backtest", use_container_width=True)
with c3:
    btn_mt5 = st.button("🔌 MT5", use_container_width=True)
with c4:
    btn_sim = st.button("🎮 Simulation", use_container_width=True)
with c5:
    btn_refresh = st.button("🔄 Refresh", use_container_width=True)

if btn_refresh:
    st.cache_data.clear(); st.success("✅ Cache vidé")

# ══════════════════════════════════════════════════════════
# MT5 PANEL
# ══════════════════════════════════════════════════════════

if btn_mt5:
    st.session_state.show_mt5 = not st.session_state.show_mt5

if st.session_state.show_mt5:
    st.divider(); st.header("🔌 AvaTrade — MT5")
    if not HAS_MT5:
        st.error("❌ MetaTrader5 non installé (Windows uniquement)")
    else:
        if not st.session_state.mt5_connected:
            mt5c = config.get("mt5", {})
            c1, c2, c3 = st.columns(3)
            with c1: login = st.text_input("Login", value=mt5c.get("login", ""), key="mt5l")
            with c2: pwd = st.text_input("Password", type="password", key="mt5p")
            with c3: srv = st.text_input("Serveur", value=mt5c.get("server", "Ava-Demo"), key="mt5s")
            if st.button("✅ CONNECTER", type="primary", key="mt5_conn"):
                ok, res = mt5_connect(login, pwd, srv)
                if ok:
                    st.session_state.mt5_connected = True
                    config["mt5"] = {"login": login, "server": srv}; save_config()
                    st.success(f"✅ Connecté ! {res.balance} {res.currency} | x{res.leverage}")
                    st.rerun()
                else:
                    st.error(f"❌ {res}")
        else:
            acc = mt5_account_info()
            if acc:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Balance", f"{round(acc['balance'], 2)} {acc['currency']}")
                c2.metric("Equity", f"{round(acc['equity'], 2)}")
                c3.metric("P&L", f"{round(acc['profit'], 2)}", delta_color="normal" if acc['profit'] >= 0 else "inverse")
                c4.metric("Mode", acc["mode"])
            if st.button("🔌 Déconnecter", key="mt5_disc"):
                mt5_disconnect(); st.rerun()
            st.markdown("#### 📈 Ouvrir position")
            o1, o2, o3 = st.columns(3)
            with o1: mt5_act = st.selectbox("Actif", list(ACTIFS.keys()), key="mt5a")
            with o2: mt5_dir = st.radio("Dir", ["ACHAT", "VENTE"], horizontal=True, key="mt5d")
            with o3: mt5_lots = st.number_input("Lots", 0.01, 10.0, 0.01, 0.01, key="mt5lo")
            o4, o5 = st.columns(2)
            with o4: mt5_sl = st.number_input("SL (0=aucun)", 0.0, step=0.1, key="mt5sl")
            with o5: mt5_tp = st.number_input("TP (0=aucun)", 0.0, step=0.1, key="mt5tp")
            if st.button("🚀 OUVRIR", type="primary", key="mt5_open"):
                ok, res = mt5_open_trade(ACTIFS[mt5_act], mt5_dir, mt5_lots, mt5_sl if mt5_sl > 0 else None, mt5_tp if mt5_tp > 0 else None)
                if ok: st.success(f"✅ #{res['ticket']} | {res['direction']} @ {round(res['price'], 5)}")
                else: st.error(f"❌ {res}")
            st.markdown("#### 📊 Positions")
            positions = mt5_get_positions()
            if positions:
                for p in positions:
                    pc1, pc2, pc3, pc4 = st.columns([2.5, 1.5, 1.5, 0.8])
                    with pc1: st.markdown(f"**{'📈' if p['type'] == 'ACHAT' else '📉'} {p['symbol']}** {p['volume']}L"); st.caption(f"#{p['ticket']}")
                    with pc2: st.metric("Prix", f"{round(p['price_current'], 5)}")
                    with pc3: st.metric("P&L", f"{round(p['profit'], 2)}", delta_color="normal" if p['profit'] >= 0 else "inverse")
                    with pc4:
                        if st.button("❌", key=f"mt5c_{p['ticket']}"):
                            mt5_close_trade(p["ticket"]); st.rerun()
            else:
                st.info("Aucune position")
            st.markdown("#### 🤖 Auto-trading")
            at = st.toggle("Ordres auto sur signaux", key="mt5_auto_t")
            if at:
                at_lots = st.number_input("Lots auto", 0.01, 1.0, 0.01, 0.01, key="mt5_at_l")
                st.session_state.mt5_auto_config = {"enabled": True, "lots": at_lots, "use_sl_tp": True}
            else:
                st.session_state.mt5_auto_config = {"enabled": False}

# ══════════════════════════════════════════════════════════
# SCAN PRINCIPAL
# ══════════════════════════════════════════════════════════

if btn_scan:
    if not actifs_choisis:
        st.warning("Sélectionne au moins un actif")
    else:
        with st.spinner("🧠 Analyse v7.0..."):
            divergences = analyser_divergences_metals()
            resultats = lancer_scan(actifs_choisis, macro_data, divergences)
            st.session_state.derniers_resultats = resultats
            st.session_state.scan_effectue = True
            st.session_state.dernieres_divergences = divergences
            alertes = [r for r in resultats if r["action"] in ["ACHAT", "VENTE"]]
            for a in alertes:
                if alert_mode == "Telegram" and tg_token and tg_chat:
                    envoyer_telegram(format_telegram(a), tg_token, tg_chat)
                elif alert_mode == "Email Bluewin" and email_addr and email_pass:
                    d = "LONG" if a['action'] == "ACHAT" else "SHORT"
                    envoyer_email(f"🚨 {d} — {a['nom']}", f"Prix: {round(a['prix'], 2)}\nScore: {round(max(a['score_achat'], a['score_vente']), 1)}", email_addr, email_pass)
                if st.session_state.mt5_auto_config.get("enabled") and st.session_state.mt5_connected:
                    cfg = st.session_state.mt5_auto_config
                    sl = a["sl_tp"]["stop_loss"] if a.get("sl_tp") and cfg.get("use_sl_tp") else None
                    tp = a["sl_tp"]["take_profit"] if a.get("sl_tp") and cfg.get("use_sl_tp") else None
                    mt5_open_trade(a["ticker"], a["action"], cfg["lots"], sl, tp)
                st.session_state.historique_signaux.append({"time": datetime.now(pytz.timezone("Europe/Zurich")).strftime("%H:%M"), "nom": a["nom"], "action": a["action"], "score": round(max(a["score_achat"], a["score_vente"]), 1)})
        st.rerun()

# ══════════════════════════════════════════════════════════
# AFFICHAGE RÉSULTATS
# ══════════════════════════════════════════════════════════

if st.session_state.scan_effectue and st.session_state.derniers_resultats:
    resultats = st.session_state.derniers_resultats
    divergences = st.session_state.get("dernieres_divergences", {})

    # Divergences DXY/Or/Argent
    if divergences and "erreur" not in divergences:
        st.markdown("---"); st.subheader("🔀 Divergences Dollar / Or / Argent (10j)")
        col1, col2, col3 = st.columns(3)
        col1.metric("💵 DXY", divergences["variations"]["DXY"])
        col2.metric("🥇 Or", divergences["variations"]["Or"])
        col3.metric("🥈 Argent", divergences["variations"]["Argent"])
        d1, d2, d3 = st.columns(3)
        with d1: st.caption(f"{divergences['DXY_vs_Or']['signal']}"); st.caption(divergences['DXY_vs_Or']['detail'])
        with d2: st.caption(f"{divergences['DXY_vs_Argent']['signal']}"); st.caption(divergences['DXY_vs_Argent']['detail'])
        with d3: st.caption(f"{divergences['Or_vs_Argent']['signal']}"); st.caption(divergences['Or_vs_Argent']['detail'])

    # Résumé
    st.markdown("---"); st.subheader("📋 Résultats")
    cols = st.columns(min(len(resultats), 4))
    for i, r in enumerate(resultats):
        with cols[i % len(cols)]:
            if r["action"] == "ACHAT": st.metric(r["nom"], f"{round(r['prix'], 2)}", f"🟢 LONG ({round(r['score_achat'], 1)})")
            elif r["action"] == "VENTE": st.metric(r["nom"], f"{round(r['prix'], 2)}", f"🔴 SHORT ({round(r['score_vente'], 1)})", delta_color="inverse")
            elif r["action"] == "PLAT": st.metric(r["nom"], f"{round(r['prix'], 2)}", "😴 Plat", delta_color="off")
            else: st.metric(r["nom"], f"{round(r['prix'], 2)}", f"⏸️ ({round(max(r['score_achat'], r['score_vente']), 1)})", delta_color="off")

    # Check-list
    st.markdown("---"); st.subheader("✅ Check-list Trade")
    mode_ck = st.radio("Mode", ["🎯 Un actif", "📊 Comparatif"], horizontal=True, key="mode_ck")
    if mode_ck == "🎯 Un actif":
        noms = [r["nom"] for r in resultats]
        actif_ck = st.selectbox("Actif", noms, key="sel_ck")
        r_c = next((r for r in resultats if r["nom"] == actif_ck), None)
        if r_c:
            with st.container(border=True):
                afficher_checklist(r_c, risque_pct)
    else:
        rows = []
        for r in resultats:
            crit = calculer_checklist(r, risque_pct)
            sc_ck = sum(1 for _, ok in crit if ok)
            verdict = "🟢 GO" if sc_ck >= 6 else "🟡 PRUDENCE" if sc_ck >= 4 else "🔴 STOP"
            rows.append({"Actif": r["nom"], "Action": r["action"], "Score": round(max(r["score_achat"], r["score_vente"]), 1), "Check": f"{sc_ck}/8", "Verdict": verdict})
        st.dataframe(pd.DataFrame(rows).sort_values("Check", ascending=False), use_container_width=True, hide_index=True)

    # Détails
    st.divider(); st.subheader("📊 Détails par actif")
    for idx, r in enumerate(resultats):
        ic = "🟢" if r["action"] == "ACHAT" else "🔴" if r["action"] == "VENTE" else "😴" if r["action"] == "PLAT" else "🟡"
        lb = "LONG" if r["action"] == "ACHAT" else "SHORT" if r["action"] == "VENTE" else r["action"]
        with st.expander(f"{ic} {r['nom']} — {lb}", expanded=(r["action"] in ["ACHAT", "VENTE"])):
            if r["action"] == "ACHAT": st.success(f"🟢 LONG — {round(r['score_achat'], 1)}/{round(SCORE_MAX, 0)}")
            elif r["action"] == "VENTE": st.error(f"🔴 SHORT — {round(r['score_vente'], 1)}/{round(SCORE_MAX, 0)}")
            elif r["action"] == "PLAT": st.warning(f"😴 ADX={round(r['adx'], 1)}")
            else: st.info(f"⏸️ {round(max(r['score_achat'], r['score_vente']), 1)}/{round(SCORE_MAX, 0)}")
            if r.get("ml"):
                ml = r["ml"]
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("🤖 Hausse", f"{round(ml['hausse'] * 100, 0)}%")
                mc2.metric("Accuracy", f"{round(ml['acc'] * 100, 0)}%")
                mc3.metric("Direction", ml["dir"])
                if ml.get("top_features"):
                    st.caption("Top: " + ", ".join([f[0] for f in ml["top_features"][:3]]))
            if r.get("sl_tp") and r["action"] in ["ACHAT", "VENTE"]:
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🛑 SL", f"{round(r['sl_tp']['stop_loss'], 2)}", f"-{round(r['sl_tp']['risque_pct'], 2)}%", delta_color="inverse")
                sc2.metric("🎯 TP", f"{round(r['sl_tp']['take_profit'], 2)}", f"+{round(r['sl_tp']['reward_pct'], 2)}%")
                sc3.metric("R:R", f"1:{round(r['sl_tp']['ratio_rr'], 1)}")
                if capital > 0:
                    nb, taille = calculer_taille_position(capital, risque_pct, r['prix'], r['sl_tp']['stop_loss'])
                    st.caption(f"💰 {capital} CHF ({risque_pct}%) → {round(nb, 4)} unités ({round(taille, 2)} CHF)")
            if r["action"] in ["ACHAT", "VENTE"]:
                if r.get("heure_avoid"): st.warning(f"⛔ Mauvaise heure ({r['heure']}h)")
                elif (r["action"] == "ACHAT" and r.get("heure_ok_buy")) or (r["action"] == "VENTE" and r.get("heure_ok_sell")): st.success(f"✅ Bonne heure ({r['heure']}h)")
                else:
                    ideal = r.get("buy_txt") if r["action"] == "ACHAT" else r.get("sell_txt")
                    st.info(f"🕐 Neutre ({r['heure']}h) | Idéal: {ideal}")
            mtf = r.get("mtf")
            if mtf and r["action"] in ["ACHAT", "VENTE"]:
                if mtf.get('consensus') == r["action"]: st.success(f"📊 MTF CONFIRME {lb}")
                elif mtf.get('consensus') != "NEUTRE": st.warning(f"⚠️ MTF = {mtf['consensus']} → CONFLIT")
            fig = graphique(r["data"], r["nom"], r.get("supports"), r.get("resistances"))
            st.plotly_chart(fig, use_container_width=True, key=f"g_{idx}")
            for ind, sig, txt in r["details"]:
                if sig == "ACHAT": st.write(f"✅ **{ind}** — {txt}")
                elif sig == "VENTE": st.write(f"❌ **{ind}** — {txt}")
                elif sig == "OK": st.write(f"💪 **{ind}** — {txt}")
                elif sig == "PLAT": st.write(f"😴 **{ind}** — {txt}")
                else: st.write(f"⏸️ **{ind}** — {txt}")

elif not st.session_state.scan_effectue:
    st.info("👆 Clique sur **SCANNER** pour lancer l'analyse")

# ══════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════

if btn_backtest:
    if not actifs_choisis:
        st.warning("Choisis au moins un actif")
    else:
        st.divider(); st.header("📊 Backtesting 1 an")
        st.caption(f"Seuil={seuil} | SL=1.5×ATR | TP=2.5×ATR | Ichimoku inclus")
        for nom in actifs_choisis:
            ticker = ACTIFS[nom]
            with st.spinner(f"Backtest {nom}..."):
                bt = backtester(ticker, seuil)
            if bt:
                with st.expander(f"📊 {nom}", expanded=True):
                    ca, cb, cc, cd, ce = st.columns(5)
                    ca.metric("P&L", f"{round(bt['pnl'], 2)}%", f"{bt['nb']} trades")
                    cb.metric("Win Rate", f"{round(bt['wr'], 1)}%", f"{bt['gagnants']}W/{bt['perdants']}L")
                    cc.metric("Profit Factor", f"{round(bt['pf'], 2)}")
                    cd.metric("Max DD", f"{round(bt['dd'], 2)}%", delta_color="inverse")
                    ce.metric("Sharpe", f"{round(bt['sharpe'], 2)}")
                    if bt['wr'] >= 60 and bt['pf'] >= 1.5: st.success("✅ PROFITABLE")
                    elif bt['wr'] >= 50: st.info("🟡 Correcte")
                    else: st.error("❌ Non rentable")
                    cumul = bt['trades']['pnl'].cumsum()
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(y=cumul.values, mode='lines', line=dict(color='#667eea', width=2)))
                    fig_eq.update_layout(title="Equity Curve", height=200, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_eq, use_container_width=True, key=f"eq_{nom}")
            else:
                st.info(f"{nom} : pas assez de données")

# ══════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════

if btn_sim:
    st.session_state.show_sim = True

if st.session_state.show_sim:
    st.divider(); st.header("🎮 Simulation (Levier x1→x200 + Frais nuit)")
    simulations = charger_json(SIMULATION_FILE)
    with st.expander("➕ Nouvelle simulation", expanded=not simulations):
        s1, s2, s3 = st.columns(3)
        with s1: sim_act = st.selectbox("Actif", list(ACTIFS.keys()), key="sa")
        with s2: sim_mnt = st.number_input("Montant CHF", 10, 1000000, 1000, 100, key="sm")
        with s3: sim_dir = st.radio("Dir", ["📈 LONG", "📉 SHORT"], key="sd")
        s4, s5 = st.columns(2)
        with s4:
            sim_lev = st.slider("⚡ Levier", 1, 200, 1, key="sl_sim")
            if sim_lev > 1:
                st.caption(f"Expo: {sim_mnt * sim_lev:,.0f} CHF | Liq: ~{round(100 / sim_lev, 1)}%")
                if sim_lev >= 50: st.warning("🚨 Levier très élevé !")
        with s5:
            sim_fee = st.number_input("🌙 Frais nuit %/j", 0.0, 1.0, 0.02, 0.005, format="%.3f", key="sf")
        if st.button("🚀 Investir (simulation)", type="primary", key="bsi"):
            px = get_prix_actuel(ACTIFS[sim_act])
            if px:
                d = "LONG" if "LONG" in sim_dir else "SHORT"
                now = datetime.now(pytz.timezone("Europe/Zurich"))
                simulations.append({"id": hashlib.md5(f"{sim_act}{now.isoformat()}".encode()).hexdigest()[:8], "actif": sim_act, "ticker": ACTIFS[sim_act], "dir": d, "montant": sim_mnt, "prix_e": px, "date": now.strftime("%d.%m.%Y %H:%M:%S"), "statut": "OPEN", "levier": sim_lev, "frais": sim_fee})
                sauver_json(SIMULATION_FILE, simulations); st.success(f"✅ {sim_act} {d} @ {round(px, 4)} x{sim_lev}"); st.rerun()
            else: st.error("Prix indisponible")

    opens = [s for s in simulations if s["statut"] == "OPEN"]
    if opens:
        st.subheader(f"📊 {len(opens)} position(s)")
        tot_inv = 0; tot_pnl = 0
        for sim in opens:
            px = get_prix_actuel(sim["ticker"])
            if not px: continue
            lev = sim.get("levier", 1); fee = sim.get("frais", 0)
            if sim["dir"] == "LONG": pnl_b = ((px - sim["prix_e"]) / sim["prix_e"]) * 100
            else: pnl_b = ((sim["prix_e"] - px) / sim["prix_e"]) * 100
            pnl_b *= lev
            try: de = pytz.timezone("Europe/Zurich").localize(datetime.strptime(sim["date"], "%d.%m.%Y %H:%M:%S"))
            except: de = datetime.now(pytz.timezone("Europe/Zurich"))
            jours = max(0, (datetime.now(pytz.timezone("Europe/Zurich")) - de).days)
            frais_chf = sim["montant"] * (fee / 100) * jours * lev
            pnl_chf = sim["montant"] * (pnl_b / 100) - frais_chf
            tot_inv += sim["montant"]; tot_pnl += pnl_chf
            if pnl_chf <= -sim["montant"]:
                st.error(f"💀 {sim['actif']} LIQUIDÉ"); sim["statut"] = "CLOSED"; sim["pnl"] = -100
                sauver_json(SIMULATION_FILE, simulations); continue
            p1, p2, p3, p4 = st.columns([2, 1.5, 1.5, 0.7])
            with p1:
                lt = f" x{lev}" if lev > 1 else ""
                st.markdown(f"**{'📈' if sim['dir'] == 'LONG' else '📉'} {sim['actif']}** ({sim['dir']}{lt})")
                if jours > 0 and fee > 0: st.caption(f"🌙 -{round(frais_chf, 2)} CHF ({jours}j)")
            with p2: st.metric("P&L", f"{round(pnl_chf, 2)} CHF", f"{'+' if pnl_b >= 0 else ''}{round(pnl_b, 1)}%", delta_color="normal" if pnl_chf >= 0 else "inverse")
            with p3: st.metric("Valeur", f"{round(sim['montant'] + pnl_chf, 2)} CHF")
            with p4:
                if st.button("❌", key=f"cs_{sim['id']}"):
                    sim["statut"] = "CLOSED"; sim["pnl"] = round(pnl_b, 2); sim["pnl_chf"] = round(pnl_chf, 2)
                    sauver_json(SIMULATION_FILE, simulations); st.rerun()
        st.metric("📈 P&L Total", f"{round(tot_pnl, 2)} CHF")

    closed = [s for s in simulations if s["statut"] == "CLOSED"]
    if closed:
        with st.expander(f"📜 Historique ({len(closed)})"):
            for sim in reversed(closed[-15:]):
                emoji = "🟢" if sim.get('pnl_chf', sim.get('pnl', 0)) >= 0 else "🔴"
                st.caption(f"{emoji} {sim['actif']} ({sim['dir']} x{sim.get('levier', 1)}) | {round(sim.get('pnl_chf', sim.get('pnl', 0)), 2)} CHF")
    if simulations:
        if st.button("🗑️ Reset simulations", key="rs_sim"):
            sauver_json(SIMULATION_FILE, []); st.rerun()

# ══════════════════════════════════════════════════════════
# ALERTES PAR PRIX CIBLE
# ══════════════════════════════════════════════════════════

st.divider(); st.header("🚨 Alertes prix cible")
alertes_prix = charger_json(ALERTES_PRIX_FILE)

with st.expander("➕ Nouvelle alerte", expanded=not alertes_prix):
    al1, al2, al3 = st.columns(3)
    with al1: al_actif = st.selectbox("Actif", list(ACTIFS.keys()), key="al_a")
    with al2:
        px_al = get_prix_actuel(ACTIFS[al_actif])
        st.caption(f"Actuel: {round(px_al, 4) if px_al else '?'}")
        al_prix = st.number_input("Prix cible", 0.0, value=float(round(px_al, 2)) if px_al else 0.0, step=0.01, key="al_p")
    with al3: al_cond = st.radio("Quand...", ["Au-dessus ⬆️", "En-dessous ⬇️"], key="al_c")
    if st.button("➕ Créer", type="primary", key="al_btn"):
        cond = "au-dessus" if "dessus" in al_cond else "en-dessous"
        alertes_prix.append({"id": hashlib.md5(f"{al_actif}{al_prix}{datetime.now().isoformat()}".encode()).hexdigest()[:8], "actif": al_actif, "ticker": ACTIFS[al_actif], "prix_cible": al_prix, "condition": cond, "date": datetime.now(pytz.timezone("Europe/Zurich")).strftime("%d.%m.%Y %H:%M"), "ok": False})
        sauver_json(ALERTES_PRIX_FILE, alertes_prix); st.success(f"✅ {al_actif} {cond} {al_prix}"); st.rerun()

actives = [a for a in alertes_prix if not a.get("ok")]
if actives:
    st.subheader(f"🔔 {len(actives)} alerte(s)")
    for al in actives:
        px = get_prix_actuel(al["ticker"])
        if not px: continue
        triggered = (al["condition"] == "au-dessus" and px >= al["prix_cible"]) or (al["condition"] == "en-dessous" and px <= al["prix_cible"])
        if triggered:
            al["ok"] = True; sauver_json(ALERTES_PRIX_FILE, alertes_prix)
            st.success(f"🚨 {al['actif']} = {round(px, 4)} ({al['condition']} {al['prix_cible']})")
            if alert_mode == "Telegram" and tg_token and tg_chat:
                envoyer_telegram(f"🚨 {al['actif']} = {round(px, 4)}\nCible: {al['condition']} {al['prix_cible']}", tg_token, tg_chat)
        else:
            dist = ((al["prix_cible"] - px) / px) * 100
            a1, a2, a3, a4 = st.columns([2, 1.5, 1.5, 0.7])
            with a1: st.markdown(f"**{'⬆️' if al['condition'] == 'au-dessus' else '⬇️'} {al['actif']}**"); st.caption(f"Créée: {al['date']}")
            with a2: st.metric("Cible", f"{al['prix_cible']}")
            with a3: st.metric("Distance", f"{round(abs(dist), 2)}%")
            with a4:
                if st.button("🗑️", key=f"da_{al['id']}"):
                    alertes_prix.remove(al); sauver_json(ALERTES_PRIX_FILE, alertes_prix); st.rerun()

if alertes_prix:
    if st.button("🗑️ Reset alertes", key="ra"):
        sauver_json(ALERTES_PRIX_FILE, []); st.rerun()

# ══════════════════════════════════════════════════════════
# NEWS + ON-CHAIN (sections basses)
# ══════════════════════════════════════════════════════════

st.divider(); st.header("📰 Sentiment News")
cn1, cn2, cn3 = st.columns(3)
with cn1:
    st.subheader("Bitcoin")
    ns, nd, nh = get_news_score("BTC-USD")
    st.metric("Score", f"{round(ns, 1)}/10"); st.caption(nd)
    for h in nh[:2]:
        e = "🟢" if h['score'] > 0.1 else "🔴" if h['score'] < -0.1 else "⚪"
        st.caption(f"{e} {h['title'][:55]}...")
with cn2:
    st.subheader("Or")
    ns2, nd2, _ = get_news_score("GC=F")
    st.metric("Score", f"{round(ns2, 1)}/10"); st.caption(nd2)
with cn3:
    st.subheader("S&P 500")
    ns3, nd3, _ = get_news_score("^GSPC")
    st.metric("Score", f"{round(ns3, 1)}/10"); st.caption(nd3)

st.divider(); st.header("⛓️ On-Chain Bitcoin")
oc_s, oc_d = get_onchain("BTC-USD")
st.metric("Score", f"{oc_s}/10")
for d in oc_d: st.write(d)

st.divider(); st.header("🔗 Or vs Bitcoin (7j)")
div_ob = get_divergence_or_btc()
if div_ob:
    do1, do2, do3 = st.columns(3)
    do1.metric("Or", f"{round(div_ob['var_or'], 1)}%")
    do2.metric("BTC", f"{round(div_ob['var_btc'], 1)}%")
    do3.metric("Écart", f"{round(div_ob['ecart'], 1)}%")

# Historique signaux
if st.session_state.historique_signaux:
    st.divider(); st.header("📜 Signaux (session)")
    for s in reversed(st.session_state.historique_signaux[-10:]):
        st.caption(f"{s['time']} | {'🟢' if s['action'] == 'ACHAT' else '🔴'} {s['nom']} — {s['score']}")

# ══════════════════════════════════════════════════════════
# AIDE
# ══════════════════════════════════════════════════════════

st.divider()
with st.expander("📖 Aide & Apprentissage"):
    tab1, tab2, tab3 = st.tabs(["🔤 Glossaire", "📊 Indicateurs", "🎓 Règles"])
    with tab1:
        st.markdown("""
| Terme | Explication |
|---|---|
| Long | Pari sur la hausse |
| Short | Pari sur la baisse |
| Stop-Loss | Limite ta perte |
| Take-Profit | Encaisse ton gain |
| R:R | 1:2 = risque 1 pour gagner 2 |
| Levier | Multiplie gains ET pertes |
| Liquidation | Position fermée de force |
| Win Rate | % trades gagnants |
| Profit Factor | Gains ÷ Pertes |
| Funding Rate | Coût positions futures |
        """)
    with tab2:
        st.markdown("""
- **RSI** < 30 = achat | > 70 = vente
- **Stoch** < 20 = achat | > 80 = vente
- **MACD** > Signal = bullish
- **Ichimoku** Au-dessus nuage = bullish
- **ADX** > 25 = tendance
- **Levier** x10 → liquidation à 10% contre toi
        """)
    with tab3:
        st.markdown("""
1. 🎮 Commence en **simulation**
2. 💰 Max **1-2%** risque par trade
3. 📐 R:R minimum **1:2**
4. 🧘 Jamais trader sous émotion
5. ⚡ Levier élevé = danger
6. 💪 Vérifie l'ADX avant d'entrer
7. 🔄 Backteste ta stratégie
8. ✅ Minimum **6/8** check-list
        """)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════

st.markdown("---")
st.caption(f"🧠 Trading Scanner v7.0 | ⚠️ Pas un conseil financier | {len(ACTIFS)} actifs | {len(POIDS)} indicateurs | Score max: {SCORE_MAX}")

# ══════════════════════════════════════════════════════════
# AUTO-SCAN (en dernier — bloque la boucle)
# ══════════════════════════════════════════════════════════

if auto_scan and actifs_choisis:
    st.subheader(f"🔄 Auto-scan — {auto_freq}")
    ph = st.empty(); ct = st.empty()
    while True:
        now = datetime.now(pytz.timezone("Europe/Zurich")).strftime("%H:%M:%S")
        with ph.container():
            st.caption(f"⏰ {now}")
            divergences = analyser_divergences_metals()
            res = lancer_scan(actifs_choisis, macro_data, divergences)
            st.session_state.derniers_resultats = res; st.session_state.scan_effectue = True
            if res:
                cols = st.columns(min(len(res), 4))
                for i, r in enumerate(res):
                    with cols[i % len(cols)]:
                        if r["action"] == "ACHAT": st.metric(r["nom"], f"{round(r['prix'], 2)}", f"🟢 {round(r['score_achat'], 1)}")
                        elif r["action"] == "VENTE": st.metric(r["nom"], f"{round(r['prix'], 2)}", f"🔴 {round(r['score_vente'], 1)}", delta_color="inverse")
                        else: st.metric(r["nom"], f"{round(r['prix'], 2)}", "⏸️", delta_color="off")
                alertes = [r for r in res if r["action"] in ["ACHAT", "VENTE"]]
                if alertes:
                    st.warning(f"🚨 {len(alertes)} signal(s)")
                    for a in alertes:
                        if alert_mode == "Telegram" and tg_token and tg_chat: envoyer_telegram(format_telegram(a), tg_token, tg_chat)
                        elif alert_mode == "Email Bluewin" and email_addr and email_pass: envoyer_email(f"🚨 {a['nom']}", f"{a['action']} @ {round(a['prix'], 2)}", email_addr, email_pass)
        for rem in range(auto_sec, 0, -1):
            m_r, s_r = divmod(rem, 60)
            ct.caption(f"⏳ {m_r:02d}:{s_r:02d}")
            time.sleep(1)
        st.rerun()
