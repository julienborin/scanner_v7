# -*- coding: utf-8 -*-
# ============================================================
# TRADING SCANNER v7.1 (CONFIRMATION + NEWS PRO + CALENDRIER + FR)
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

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Trading Scanner v7.1", page_icon="🧠", layout="wide")

# ══════════════════════════════════════════════════════════
# TRADUCTION EN FRANÇAIS
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def traduire(texte):
    """Traduit un texte anglais en français (cache 1h)"""
    if not HAS_TRANSLATOR or not texte:
        return texte
    try:
        return GoogleTranslator(source='en', target='fr').translate(texte[:200])
    except:
        return texte


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
    "GC=F": ["gold price", "XAUUSD", "precious metals", "gold rally", "central bank gold", "gold reserves", "gold ETF", "gold demand"],
    "SI=F": ["silver price", "XAGUSD", "silver demand", "silver ETF", "industrial silver"],
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
    "PL=F": ["platinum price", "platinum demand", "platinum ETF"],
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
    if not HAS_MT5: return False, "MetaTrader5 non installé"
    if not mt5.initialize(): return False, f"Erreur init: {mt5.last_error()}"
    if not mt5.login(login=int(login), password=password, server=server):
        mt5.shutdown(); return False, f"Login échoué: {mt5.last_error()}"
    info = mt5.account_info()
    if not info: mt5.shutdown(); return False, "Infos indisponibles"
    return True, info


def mt5_disconnect():
    if HAS_MT5: mt5.shutdown()
    st.session_state.mt5_connected = False


def mt5_account_info():
    if not HAS_MT5 or not st.session_state.mt5_connected: return None
    try:
        info = mt5.account_info()
        if info: return {"balance": info.balance, "equity": info.equity, "profit": info.profit, "margin_free": info.margin_free, "currency": info.currency, "leverage": info.leverage, "login": info.login, "server": info.server, "mode": "DEMO" if info.trade_mode == 0 else "RÉEL"}
    except: pass
    return None


def mt5_open_trade(ticker, direction, lots, sl=None, tp=None):
    if not HAS_MT5 or not st.session_state.mt5_connected: return False, "Non connecté"
    symbol = MT5_SYMBOLS.get(ticker)
    if not symbol: return False, f"{ticker} non mappé"
    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        for sfx in [".ava", "_ava", ""]:
            sym_info = mt5.symbol_info(symbol + sfx)
            if sym_info: symbol = symbol + sfx; break
    if not sym_info: return False, f"{symbol} introuvable"
    if not sym_info.visible: mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return False, "Pas de cotation"
    if direction == "ACHAT": otype = mt5.ORDER_TYPE_BUY; price = tick.ask
    else: otype = mt5.ORDER_TYPE_SELL; price = tick.bid
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lots), "type": otype, "price": price, "deviation": 20, "magic": 700000, "comment": "ScannerV7", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    if sl and sl > 0: req["sl"] = float(sl)
    if tp and tp > 0: req["tp"] = float(tp)
    result = mt5.order_send(req)
    if not result: return False, f"Erreur: {mt5.last_error()}"
    if result.retcode != mt5.TRADE_RETCODE_DONE: return False, f"Rejeté [{result.retcode}]: {result.comment}"
    return True, {"ticket": result.order, "symbol": symbol, "direction": direction, "lots": lots, "price": price}


def mt5_close_trade(ticket):
    if not HAS_MT5: return False, "MT5 indisponible"
    pos = mt5.positions_get(ticket=ticket)
    if not pos: return False, f"#{ticket} introuvable"
    p = pos[0]
    if p.type == mt5.ORDER_TYPE_BUY: otype = mt5.ORDER_TYPE_SELL; price = mt5.symbol_info_tick(p.symbol).bid
    else: otype = mt5.ORDER_TYPE_BUY; price = mt5.symbol_info_tick(p.symbol).ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "type": otype, "position": ticket, "price": price, "deviation": 20, "magic": 700000, "comment": "V7 Close", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    result = mt5.order_send(req)
    if not result: return False, f"Erreur: {mt5.last_error()}"
    if result.retcode != mt5.TRADE_RETCODE_DONE: return False, f"Rejeté [{result.retcode}]"
    return True, {"ticket": ticket, "profit": p.profit}


def mt5_get_positions():
    if not HAS_MT5 or not st.session_state.mt5_connected: return []
    try:
        positions = mt5.positions_get()
        if not positions: return []
        return [{"ticket": p.ticket, "symbol": p.symbol, "type": "ACHAT" if p.type == 0 else "VENTE", "volume": p.volume, "price_open": p.price_open, "price_current": p.price_current, "sl": p.sl, "tp": p.tp, "profit": p.profit, "swap": p.swap, "time": datetime.fromtimestamp(p.time).strftime("%d.%m %H:%M")} for p in positions]
    except: return []


# ══════════════════════════════════════════════════════════
# TELEGRAM + EMAIL
# ══════════════════════════════════════════════════════════

def envoyer_telegram(message, token, chat_id):
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except: return False


def format_telegram(r):
    emoji = "🟢" if r["action"] == "ACHAT" else "🔴"
    d = "LONG" if r["action"] == "ACHAT" else "SHORT"
    sc = r["score_achat"] if r["action"] == "ACHAT" else r["score_vente"]
    msg = f"{emoji} <b>{d}</b> — {r['nom']}\n💰 {round(r['prix'], 2)} | Score: {round(sc, 1)}/{round(SCORE_MAX, 0)}"
    if r.get("sl_tp"): msg += f"\n🛑 SL: {round(r['sl_tp']['stop_loss'], 2)} | 🎯 TP: {round(r['sl_tp']['take_profit'], 2)} | R:R 1:{round(r['sl_tp']['ratio_rr'], 1)}"
    if r.get("ml") and r["ml"].get("acc", 0) > 0.52: msg += f"\n🤖 ML: {round(r['ml']['hausse'] * 100, 0)}% (acc {round(r['ml']['acc'] * 100, 0)}%)"
    if r.get("divergences_txt"): msg += f"\n🔀 {r['divergences_txt']}"

    # --- CONTEXTE MACRO + NEWS ---
    try:
        macro_d, _ = fetch_macro()
        cat = ACTIF_CATEGORIE.get(r.get("nom", ""), "forex")
        m_score = calc_macro_score(macro_d, cat)
        if m_score >= 3: msg += f"\n🌍 Macro: 🟢 +{round(m_score, 1)}/10"
        elif m_score <= -3: msg += f"\n🌍 Macro: 🔴 {round(m_score, 1)}/10"
        else: msg += f"\n🌍 Macro: ⚖️ {round(m_score, 1)}/10"
    except: pass

    try:
        n_score, _, _ = get_news_score(r.get("ticker", ""))
        if n_score >= 3: msg += f"\n📰 News: 🟢 +{round(n_score, 1)}/10"
        elif n_score <= -3: msg += f"\n📰 News: 🔴 {round(n_score, 1)}/10"
    except: pass

    try:
        hi = check_high_impact_event()
        if hi: msg += f"\n⚠️ ATTENTION: {len(hi)} événement(s) macro à venir!"
    except: pass

    # Heure idéale
    heure_ok = (r.get("heure_ok_buy") and r["action"] == "ACHAT") or (r.get("heure_ok_sell") and r["action"] == "VENTE")
    msg += f"\n⏰ {datetime.now(pytz.timezone('Europe/Zurich')).strftime('%H:%M:%S')} {'✅' if heure_ok else '⚠️ Hors créneau idéal'}"
    return msg



def envoyer_email(sujet, message, email_addr, email_pass):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = sujet; msg['From'] = email_addr; msg['To'] = email_addr
        msg.attach(MIMEText(message, 'plain', 'utf-8'))
        with smtplib.SMTP_SSL("smtpauths.bluewin.ch", 465) as server:
            server.login(email_addr, email_pass); server.send_message(msg)
        return True
    except Exception as e: return str(e)


# ══════════════════════════════════════════════════════════
# DATA — TÉLÉCHARGEMENT
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_ccxt(symbol, tf="1d", limit=365):
    if not HAS_CCXT: return None
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms"); df.set_index("ts", inplace=True)
        return df
    except: return None


@st.cache_data(ttl=30)
def fetch_orderbook(symbol):
    if not HAS_CCXT: return None
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        ob = ex.fetch_order_book(symbol, limit=50)
        bids = sum(b[1] for b in ob["bids"][:20]); asks = sum(a[1] for a in ob["asks"][:20])
        total = bids + asks
        if total == 0: return None
        return {"imbalance": (bids - asks) / total, "bids": bids, "asks": asks}
    except: return None


@st.cache_data(ttl=300, show_spinner="📥 Données...")
def telecharger(ticker):
    if HAS_CCXT and ticker in CCXT_SYMBOLS:
        data = fetch_ccxt(CCXT_SYMBOLS[ticker], "1d", 365)
        if data is not None and not data.empty: return data
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=15)
    except Exception:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    # Fix MultiIndex (yfinance récent)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Fix colonnes dupliquées (peut arriver après le flatten)
    data = data.loc[:, ~data.columns.duplicated()]
    return data



@st.cache_data(ttl=300)
def telecharger_4h(ticker):
    if HAS_CCXT and ticker in CCXT_SYMBOLS:
        data = fetch_ccxt(CCXT_SYMBOLS[ticker], "4h", 200)
        if data is not None and not data.empty: return data
    data = yf.download(ticker, period="60d", interval="1h", progress=False)
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    if data.empty: return data
    return data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()


@st.cache_data(ttl=300)
def telecharger_weekly(ticker):
    data = yf.download(ticker, period="2y", interval="1wk", progress=False)
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    return data


def get_prix_actuel(ticker):
    try:
        if HAS_CCXT and ticker in CCXT_SYMBOLS:
            ex = ccxt.binance({"enableRateLimit": True}); t = ex.fetch_ticker(CCXT_SYMBOLS[ticker])
            if t and t.get("last"): return float(t["last"])
    except: pass
    try:
        t = yf.Ticker(ticker); p = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
        if p and p > 0: return float(p)
    except: pass
    try:
        d = yf.download(ticker, period="5d", interval="1d", progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        if not d.empty: return float(d["Close"].iloc[-1])
    except: pass
    return None


# ══════════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ══════════════════════════════════════════════════════════

def calc_rsi(s, p=14):
    if HAS_PANDAS_TA:
        r = ta.rsi(s, length=p)
        if r is not None: return r
    delta = s.diff(); gain = delta.where(delta > 0, 0).rolling(p).mean(); loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + gain / loss))


def calc_stoch(data, p=14):
    if HAS_PANDAS_TA:
        r = ta.stoch(data["High"], data["Low"], data["Close"], k=p)
        if r is not None and not r.empty: c = r.columns.tolist(); return r[c[0]], r[c[1]]
    lo = data["Low"].rolling(p).min(); hi = data["High"].rolling(p).max()
    k = ((data["Close"] - lo) / (hi - lo)) * 100
    return k, k.rolling(3).mean()


def calc_macd(close):
    if HAS_PANDAS_TA:
        r = ta.macd(close, fast=12, slow=26, signal=9)
        if r is not None and not r.empty: c = r.columns.tolist(); return r[c[0]], r[c[2]]
    e12 = close.ewm(span=12).mean(); e26 = close.ewm(span=26).mean(); m = e12 - e26
    return m, m.ewm(span=9).mean()


def calc_adx(data, p=14):
    if HAS_PANDAS_TA:
        r = ta.adx(data["High"], data["Low"], data["Close"], length=p)
        if r is not None and not r.empty: return r.iloc[:, 0]
    h = data["High"]; l = data["Low"]; c = data["Close"]
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    pdm = h.diff().where(h.diff() > -l.diff(), 0).where(h.diff() > 0, 0)
    mdm = (-l.diff()).where(-l.diff() > h.diff(), 0).where(-l.diff() > 0, 0)
    atr = tr.rolling(p).mean(); pdi = 100 * pdm.rolling(p).mean() / atr; mdi = 100 * mdm.rolling(p).mean() / atr
    dx = 100 * abs(pdi - mdi) / (pdi + mdi)
    return dx.rolling(p).mean()


def calc_atr(data, p=14):
    if HAS_PANDAS_TA:
        r = ta.atr(data["High"], data["Low"], data["Close"], length=p)
        if r is not None: return r
    h = data["High"]; l = data["Low"]; c = data["Close"]
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.rolling(p).mean()


def calc_bollinger(close, p=20, m=2):
    if HAS_PANDAS_TA:
        r = ta.bbands(close, length=p, std=m)
        if r is not None and not r.empty: c = r.columns.tolist(); return r[c[2]], r[c[0]], r[c[1]]
    sma = close.rolling(p).mean(); std = close.rolling(p).std()
    return sma + m * std, sma - m * std, sma


def calc_ichimoku(data):
    h = data["High"]; l = data["Low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spa = ((tenkan + kijun) / 2).shift(26); spb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, spa, spb


def calc_vwap(data):
    if HAS_PANDAS_TA:
        r = ta.vwap(data["High"], data["Low"], data["Close"], data["Volume"])
        if r is not None: return r
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    return (tp * data["Volume"]).cumsum() / data["Volume"].cumsum()


def calc_fibonacci(data, p=50):
    hi = data["High"].rolling(p).max(); lo = data["Low"].rolling(p).min()
    return hi - 0.618 * (hi - lo), hi - 0.382 * (hi - lo)


def detect_supports_resistances(data, window=20):
    close = data["Close"].values; high = data["High"].values; low = data["Low"].values
    supports = []; resistances = []
    for i in range(window, len(close) - window):
        if low[i] == min(low[i - window:i + window + 1]): supports.append(low[i])
        if high[i] == max(high[i - window:i + window + 1]): resistances.append(high[i])
    def cluster(lvls):
        if not lvls: return []
        lvls = sorted(lvls); cl = [lvls[0]]
        for lv in lvls[1:]:
            if (lv - cl[-1]) / cl[-1] > 0.02: cl.append(lv)
            else: cl[-1] = (cl[-1] + lv) / 2
        return cl
    supports = cluster(supports); resistances = cluster(resistances)
    px = close[-1]
    supports = sorted(supports, key=lambda x: abs(x - px))[:5]
    resistances = sorted(resistances, key=lambda x: abs(x - px))[:5]
    return sorted(supports), sorted(resistances)


def detecter_divergences_tech(data, lookback=14):
    result = {'rsi': None, 'macd': None}
    if len(data) < lookback + 5: return result
    close = data['Close'].values; rsi = data['RSI'].values; macd = data['MACD'].values
    try:
        recent = close[-lookback:]; recent_rsi = rsi[-lookback:]; recent_macd = macd[-lookback:]
        prix_lows = []
        for i in range(2, lookback - 2):
            if recent[i] <= min(recent[i-2:i]) and recent[i] <= min(recent[i+1:i+3]):
                prix_lows.append((i, recent[i], recent_rsi[i], recent_macd[i]))
        if len(prix_lows) >= 2:
            last = prix_lows[-1]; prev = prix_lows[-2]
            if last[1] < prev[1] and last[2] > prev[2]: result['rsi'] = "HAUSSIERE"
            if last[1] < prev[1] and last[3] > prev[3]: result['macd'] = "HAUSSIERE"
        prix_highs = []
        for i in range(2, lookback - 2):
            if recent[i] >= max(recent[i-2:i]) and recent[i] >= max(recent[i+1:i+3]):
                prix_highs.append((i, recent[i], recent_rsi[i], recent_macd[i]))
        if len(prix_highs) >= 2:
            last = prix_highs[-1]; prev = prix_highs[-2]
            if last[1] > prev[1] and last[2] < prev[2]: result['rsi'] = "BAISSIERE"
            if last[1] > prev[1] and last[3] < prev[3]: result['macd'] = "BAISSIERE"
    except: pass
    return result


def calculer_indicateurs(data):
    data["RSI"] = calc_rsi(data["Close"])
    data["Stoch_K"], data["Stoch_D"] = calc_stoch(data)
    data["MACD"], data["MACD_Signal"] = calc_macd(data["Close"])
    data["ADX"] = calc_adx(data); data["ATR"] = calc_atr(data)
    data["BB_Upper"], data["BB_Lower"], data["BB_Mid"] = calc_bollinger(data["Close"])
    data["MA_200"] = data["Close"].rolling(200).mean(); data["MA_50"] = data["Close"].rolling(50).mean()
    data["Vol_Moy"] = data["Volume"].rolling(20).mean(); data["VWAP"] = calc_vwap(data)
    data["Fib_618"], data["Fib_382"] = calc_fibonacci(data)
    tk, kj, spa, spb = calc_ichimoku(data)
    data["Ichi_TK"] = tk; data["Ichi_KJ"] = kj; data["Ichi_SpA"] = spa; data["Ichi_SpB"] = spb
    return data


# ══════════════════════════════════════════════════════════
# CALENDRIER ÉCONOMIQUE USA
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner="📅 Calendrier éco...")
def get_economic_calendar():
    events = []
    vader = SentimentIntensityAnalyzer()
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Fed+decision+OR+CPI+OR+NFP+OR+FOMC+OR+inflation+data+OR+jobs+report+OR+rate+decision+when:3d&hl=en&gl=US&ceid=US:en")
        if feed.entries:
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                tl = title.lower()
                upcoming_kw = ["upcoming", "ahead", "preview", "expect", "forecast", "tomorrow", "today", "this week", "scheduled", "watch", "awaits", "brace", "prepare"]
                is_upcoming = any(k in tl for k in upcoming_kw)
                importance = "⚪"
                if any(k in tl for k in ["fed", "fomc", "rate decision", "powell", "federal reserve"]):
                    importance = "🔴"
                elif any(k in tl for k in ["cpi", "inflation", "nfp", "jobs report", "employment", "payroll"]):
                    importance = "🟠"
                elif any(k in tl for k in ["gdp", "retail sales", "pmi", "manufacturing", "consumer confidence"]):
                    importance = "🟡"
                if importance != "⚪":
                    score = vader.polarity_scores(title)["compound"]
                    pub_date = entry.get("published", "")
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub_date)
                        date_str = dt.strftime("%d.%m %H:%M")
                    except:
                        date_str = pub_date[:16] if pub_date else "?"
                    events.append({"title": title, "importance": importance, "upcoming": is_upcoming, "score": score, "date": date_str, "date_raw": date_str})
    except:
        pass
    return events



def check_high_impact_event():
    events = get_economic_calendar()
    return [e for e in events if e.get("importance") in ["🔴", "🟠"] and e.get("upcoming")]


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
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        if len(dxy) < 10 or len(or_data) < 10 or len(argent_data) < 10: return {"erreur": "Pas assez de données"}
        var_dxy = float((dxy['Close'].iloc[-1] - dxy['Close'].iloc[-10]) / dxy['Close'].iloc[-10] * 100)
        var_or = float((or_data['Close'].iloc[-1] - or_data['Close'].iloc[-10]) / or_data['Close'].iloc[-10] * 100)
        var_ag = float((argent_data['Close'].iloc[-1] - argent_data['Close'].iloc[-10]) / argent_data['Close'].iloc[-10] * 100)
        res = {}
        if var_dxy > 0 and var_or > 0: res["DXY_vs_Or"] = {"signal": "🟢 FORCE", "detail": f"DXY ({var_dxy:+.1f}%) + Or ({var_or:+.1f}%) → Force Or", "impact_or": "renforce_achat"}
        elif var_dxy < 0 and var_or < 0: res["DXY_vs_Or"] = {"signal": "🔴 FAIBLESSE", "detail": f"DXY ({var_dxy:+.1f}%) + Or ({var_or:+.1f}%) → Faiblesse", "impact_or": "renforce_vente"}
        else: res["DXY_vs_Or"] = {"signal": "⚪ Normal", "detail": f"DXY ({var_dxy:+.1f}%) vs Or ({var_or:+.1f}%)", "impact_or": "neutre"}
        if var_dxy > 0 and var_ag > 0: res["DXY_vs_Argent"] = {"signal": "🟢 FORCE", "detail": "DXY+Ag force", "impact_argent": "renforce_achat"}
        elif var_dxy < 0 and var_ag < 0: res["DXY_vs_Argent"] = {"signal": "🔴 FAIBLESSE", "detail": "DXY-Ag faiblesse", "impact_argent": "renforce_vente"}
        else: res["DXY_vs_Argent"] = {"signal": "⚪ Normal", "detail": f"DXY ({var_dxy:+.1f}%) vs Ag ({var_ag:+.1f}%)", "impact_argent": "neutre"}
        if var_or > 0 and var_ag < 0: res["Or_vs_Argent"] = {"signal": "🟡 PRUDENCE", "detail": "Or↑ sans Argent = peur", "impact_or": "affaiblit_achat", "impact_argent": "neutre"}
        elif var_or < 0 and var_ag > 0: res["Or_vs_Argent"] = {"signal": "🟢 RATTRAPAGE", "detail": "Argent rattrape", "impact_or": "neutre", "impact_argent": "renforce_achat"}
        else: res["Or_vs_Argent"] = {"signal": "⚪ Normal", "detail": "Même direction", "impact_or": "neutre", "impact_argent": "neutre"}
        res["variations"] = {"DXY": f"{var_dxy:+.1f}%", "Or": f"{var_or:+.1f}%", "Argent": f"{var_ag:+.1f}%"}
        return res
    except Exception as e: return {"erreur": str(e)}


def get_divergence_metal_impact(ticker, divergences):
    if "erreur" in divergences: return 0, ""
    if ticker == "GC=F":
        imp = divergences.get("DXY_vs_Or", {}).get("impact_or", "neutre")
        imp2 = divergences.get("Or_vs_Argent", {}).get("impact_or", "neutre")
        if imp == "renforce_achat": return 1, "DXY↑ + Or↑"
        elif imp == "renforce_vente": return -1, "DXY↓ + Or↓"
        elif imp2 == "affaiblit_achat": return -1, "Or↑ sans Ag"
        return 0, ""
    elif ticker == "SI=F":
        imp = divergences.get("DXY_vs_Argent", {}).get("impact_argent", "neutre")
        imp2 = divergences.get("Or_vs_Argent", {}).get("impact_argent", "neutre")
        if imp == "renforce_achat" or imp2 == "renforce_achat": return 1, "Force Argent"
        elif imp == "renforce_vente": return -1, "Faiblesse Ag"
        return 0, ""
    elif ticker == "PL=F":
        imp = divergences.get("DXY_vs_Argent", {}).get("impact_argent", "neutre")
        if imp == "renforce_achat": return 1, "Force métaux"
        elif imp == "renforce_vente": return -1, "Faiblesse métaux"
        return 0, ""
    return 0, ""


@st.cache_data(ttl=600)
def get_divergence_or_btc():
    try:
        g = yf.download("GC=F", period="30d", interval="1d", progress=False)
        b = yf.download("BTC-USD", period="30d", interval="1d", progress=False)
        if isinstance(g.columns, pd.MultiIndex): g.columns = g.columns.get_level_values(0)
        if isinstance(b.columns, pd.MultiIndex): b.columns = b.columns.get_level_values(0)
        if len(g) < 7 or len(b) < 7: return None
        vo = float((g["Close"].iloc[-1] - g["Close"].iloc[-7]) / g["Close"].iloc[-7] * 100)
        vb = float((b["Close"].iloc[-1] - b["Close"].iloc[-7]) / b["Close"].iloc[-7] * 100)
        return {"var_or": vo, "var_btc": vb, "ecart": vo - vb}
    except: return None


def indicateur_or_btc(ticker):
    data = get_divergence_or_btc()
    if not data: return 0, ""
    ecart = data["ecart"]
    if ticker == "GC=F":
        if ecart < -5: return 1, "BTC > Or → rattrapage"
        elif ecart > 5: return -1, "Or > BTC → excès"
    elif ticker in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        if ecart > 5: return 1, "Or > Crypto → rattrapage"
        elif ecart < -5: return -1, "Crypto > Or → excès"
    return 0, ""


# ══════════════════════════════════════════════════════════
# MACRO + NEWS + ON-CHAIN
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner="🌍 Macro...")
def fetch_macro():
    data = {}; details = []
    def dl(ticker):
        d = yf.download(ticker, period="60d", interval="1d", progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        return d
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if r.status_code == 200:
            fg = int(r.json()["data"][0]["value"]); fg_label = r.json()["data"][0]["value_classification"]
            data["fear_greed"] = {"value": fg, "score": (fg - 50) / 5, "label": fg_label}
            details.append(f"😱 Fear&Greed: {fg}/100 ({fg_label})")
    except: data["fear_greed"] = {"value": 50, "score": 0}
    try:
        vix = dl("^VIX")
        if len(vix) >= 5:
            v = float(vix["Close"].iloc[-1])
            sc = -5 if v > 30 else -3 if v > 25 else 3 if v < 15 else 1 if v < 18 else 0
            data["vix"] = {"value": v, "score": sc}; details.append(f"{'✅' if sc >= 0 else '❌'} VIX: {round(v, 1)}")
    except: data["vix"] = {"score": 0}
    try:
        dxy = dl("DX-Y.NYB")
        if len(dxy) >= 20:
            prix = float(dxy['Close'].iloc[-1]); ma20 = float(dxy['Close'].rolling(20).mean().iloc[-1])
            var_5j = float((dxy['Close'].iloc[-1] - dxy['Close'].iloc[-5]) / dxy['Close'].iloc[-5] * 100)
            sc = 0
            if prix < ma20: sc += 3
            if var_5j < -0.5: sc += 2
            elif var_5j > 0.5: sc -= 2
            if prix > ma20: sc -= 3
            data["dxy"] = {"prix": prix, "score": max(-10, min(10, sc))}; details.append(f"{'✅' if sc > 0 else '❌'} Dollar: {round(prix, 1)} ({var_5j:+.1f}%)")
    except: data["dxy"] = {"score": 0}
    try:
        tnx = dl("^TNX")
        if len(tnx) >= 20:
            prix = float(tnx['Close'].iloc[-1]); ma20 = float(tnx['Close'].rolling(20).mean().iloc[-1])
            sc = 3 if prix < ma20 else -3
            if prix > 4.5: sc -= 2
            elif prix < 3.5: sc += 2
            data["yields"] = {"prix": prix, "score": max(-10, min(10, sc))}; details.append(f"{'✅' if sc > 0 else '❌'} Taux 10Y: {round(prix, 2)}%")
    except: data["yields"] = {"score": 0}
    try:
        spy = dl("SPY")
        if len(spy) >= 20:
            prix = float(spy['Close'].iloc[-1]); ma20 = float(spy['Close'].rolling(20).mean().iloc[-1])
            var_5j = float((spy['Close'].iloc[-1] - spy['Close'].iloc[-5]) / spy['Close'].iloc[-5] * 100)
            sc = 3 if prix > ma20 else -3
            if var_5j > 2: sc += 2
            elif var_5j < -2: sc -= 2
            data["spy"] = {"prix": prix, "score": max(-10, min(10, sc))}; details.append(f"{'✅' if sc > 0 else '❌'} S&P: {var_5j:+.1f}% (5j)")
    except: data["spy"] = {"score": 0}
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": "BTCUSDT", "limit": 1}, timeout=5)
        if r.status_code == 200:
            rate = float(r.json()[0]["fundingRate"])
            sc = -3 if rate > 0.0005 else 3 if rate < -0.0001 else 0
            data["funding"] = {"current": rate * 100, "score": sc}; details.append(f"📊 Funding BTC: {round(rate * 100, 4)}%")
    except: data["funding"] = {"score": 0}
    return data, details


def calc_macro_score(macro_data, categorie):
    if categorie == "crypto": weights = {'dxy': 0.15, 'yields': 0.10, 'vix': 0.10, 'fear_greed': 0.25, 'funding': 0.20, 'spy': 0.20}
    elif categorie == "commodities": weights = {'dxy': 0.30, 'yields': 0.25, 'vix': 0.15, 'spy': 0.15, 'fear_greed': 0.15}
    elif categorie == "forex": weights = {'dxy': 0.35, 'yields': 0.25, 'vix': 0.20, 'spy': 0.20}
    else: weights = {'spy': 0.30, 'vix': 0.25, 'yields': 0.20, 'dxy': 0.10, 'fear_greed': 0.15}
    total = 0; tw = 0
    for f, w in weights.items():
        if f in macro_data and 'score' in macro_data[f]: total += macro_data[f]['score'] * w; tw += w
    sensitivity = MACRO_SENSITIVITY.get(categorie, 0.7)
    return max(-10, min(10, (total / tw * sensitivity) if tw > 0 else 0))


@st.cache_data(ttl=600, show_spinner="📰 News...")
def get_news_score(ticker):
    try:
        kw = NEWS_KEYWORDS.get(ticker, [ticker])
        q = "+".join(kw[:3]).replace(" ", "+")

        exclude = ""
        if ticker == "GC=F":
            exclude = "+-bitcoin+-crypto+-ETF+crypto"
        elif ticker == "^GSPC":
            exclude = "+-bitcoin+-crypto+-gold"
        elif ticker == "BTC-USD":
            exclude = "+-gold+price+-silver+-platinum"

        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}{exclude}+when:3d&hl=en&gl=US&ceid=US:en")
        entries = feed.entries[:15] if feed.entries else []

        if ticker in ["GC=F", "SI=F", "PL=F"]:
            try:
                kitco_feed = feedparser.parse("https://www.kitco.com/feed/rss/news/")
                if kitco_feed.entries:
                    entries.extend(kitco_feed.entries[:10])
            except:
                pass

        if not entries:
            return 0, "Pas d'articles", []

        vader = SentimentIntensityAnalyzer()
        scores = []
        headlines = []

        mots_requis = {
            "GC=F": ["gold", "or", "xau", "precious", "metal", "bullion"],
            "SI=F": ["silver", "xag", "argent"],
            "BTC-USD": ["bitcoin", "btc", "crypto"],
            "ETH-USD": ["ethereum", "eth", "defi"],
            "^GSPC": ["s&p", "wall street", "stock market", "dow", "nasdaq", "equities"],
            "AAPL": ["apple", "aapl", "iphone"],
            "MSFT": ["microsoft", "msft", "azure"],
            "TSLA": ["tesla", "tsla", "musk"],
            "CL=F": ["oil", "crude", "opec", "wti", "brent"],
            "EURUSD=X": ["eur", "usd", "euro", "dollar", "forex"],
            "EURCHF=X": ["eur", "chf", "franc", "snb"],
            "PL=F": ["platinum", "platine"],
            "SOL-USD": ["solana", "sol"],
            "LINK-USD": ["chainlink", "link"],
            "BNB-USD": ["bnb", "binance"],
            "^IXIC": ["nasdaq", "tech"],
        }

        filtre = mots_requis.get(ticker, [])
        entries_filtrees = []
        for entry in entries[:25]:
            title_lower = entry.get("title", "").lower()
            if filtre:
                if any(mot in title_lower for mot in filtre):
                    entries_filtrees.append(entry)
            else:
                entries_filtrees.append(entry)

        if not entries_filtrees:
            entries_filtrees = entries[:25]

        for entry in entries_filtrees[:25]:
            title = entry.get("title", "")
            compound = vader.polarity_scores(title)["compound"]
            tl = title.lower()
            bull = sum(1 for k in BULLISH_KW if k in tl)
            bear = sum(1 for k in BEARISH_KW if k in tl)
            final = max(-1, min(1, compound + (bull - bear) * 0.25))
            scores.append(final)

            pub_date = entry.get("published", "")
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub_date)
                date_str = dt.strftime("%d.%m %H:%M")
                date_raw = dt.timestamp()
            except:
                date_str = pub_date[:16] if pub_date else "?"
                date_raw = 0

            headlines.append({"title": title, "score": final, "date": date_str, "date_raw": date_raw})

        if scores:
            now_ts = time.time()
            filtered = [(s, h) for s, h in zip(scores, headlines) if (now_ts - h.get('date_raw', 0)) < 259200]
            if filtered:
                scores, headlines = zip(*filtered)
                scores = list(scores)
                headlines = list(headlines)

            weights = np.linspace(1.5, 0.5, len(scores))
            avg = np.average(scores, weights=weights)
            score = max(-10, min(10, avg * 10))
            bull_n = sum(1 for s in scores if s > 0.1)
            bear_n = sum(1 for s in scores if s < -0.1)
            detail = f"{bull_n}+ / {len(scores) - bull_n - bear_n}= / {bear_n}-"
            headlines_sorted = sorted(headlines, key=lambda x: x.get('date_raw', 0), reverse=True)
            return score, detail, headlines_sorted[:5]

    except:
        pass

    return 0, "Erreur", []




@st.cache_data(ttl=900)
def get_onchain(ticker):
    if ticker not in ['BTC-USD', 'ETH-USD', 'SOL-USD']: return 0, []
    score = 0; details = []
    if ticker == 'BTC-USD':
        try:
            r = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=5)
            if r.status_code == 200:
                f = r.json().get("fastestFee", 0)
                if f > 100: score -= 2; details.append(f"🔴 Fees élevés ({f} sat/vB)")
                elif f < 10: score += 1; details.append(f"🟢 Fees bas ({f} sat/vB)")
                else: details.append(f"⚪ Fees {f} sat/vB")
        except: pass
        try:
            r = requests.get("https://mempool.space/api/mempool", timeout=5)
            if r.status_code == 200:
                count = r.json().get('count', 0)
                if count > 100000: score -= 1; details.append(f"🔴 Mempool ({count} TX)")
                elif count < 10000: score += 1; details.append(f"🟢 Mempool calme ({count} TX)")
        except: pass
        try:
            r = requests.get("https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json", timeout=10)
            if r.status_code == 200:
                values = [p['y'] for p in r.json().get('values', [])]
                if len(values) >= 14:
                    change = (np.mean(values[-7:]) - np.mean(values[:7])) / np.mean(values[:7]) * 100
                    if change > 5: score += 2; details.append(f"🟢 Hashrate +{round(change, 1)}%")
                    elif change < -5: score -= 2; details.append(f"🔴 Hashrate {round(change, 1)}%")
        except: pass
        try:
            r = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio", params={"symbol": "BTCUSDT", "period": "1h", "limit": 1}, timeout=5)
            if r.status_code == 200 and r.json():
                ls = float(r.json()[0]['longShortRatio'])
                if ls > 2.0: score -= 2; details.append(f"🔴 L/S {round(ls, 2)} (trop longs)")
                elif ls < 0.8: score += 2; details.append(f"🟢 L/S {round(ls, 2)} (shorts)")
                else: details.append(f"⚪ L/S {round(ls, 2)}")
        except: pass
    try:
        r = requests.get("https://api.llama.fi/v2/historicalChainTvl", timeout=10)
        if r.status_code == 200:
            tvl_data = r.json()
            if len(tvl_data) >= 7:
                recent = tvl_data[-1].get('tvl', 0); week_ago = tvl_data[-7].get('tvl', 0)
                change = (recent - week_ago) / week_ago * 100 if week_ago > 0 else 0
                if change > 5: score += 2; details.append(f"🟢 TVL +{round(change, 1)}%")
                elif change < -5: score -= 2; details.append(f"🔴 TVL {round(change, 1)}%")
    except: pass
    return max(-10, min(10, score)), details


# ══════════════════════════════════════════════════════════
# MACHINE LEARNING
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="🤖 ML...")
def ml_predict(ticker, data):
    try:
        if len(data) < 120: return None
        df = data.copy()
        df["r1"] = df["Close"].pct_change(1); df["r5"] = df["Close"].pct_change(5); df["r10"] = df["Close"].pct_change(10)
        df["vol10"] = df["r1"].rolling(10).std(); df["mom10"] = df["Close"] / df["Close"].shift(10) - 1
        df["pma20"] = df["Close"] / df["Close"].rolling(20).mean() - 1; df["pma50"] = df["Close"] / df["Close"].rolling(50).mean() - 1
        df["vratio"] = df["Volume"] / df["Volume"].rolling(20).mean(); df["rsi_d"] = df["RSI"].diff(3)
        df["macd_h"] = df["MACD"] - df["MACD_Signal"]; df["macd_hd"] = df["macd_h"].diff(3)
        bb_r = df["BB_Upper"] - df["BB_Lower"]; df["bb_pos"] = (df["Close"] - df["BB_Lower"]) / bb_r
        df["atr_p"] = df["ATR"] / df["Close"] * 100
        df["target"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.01).astype(int)
        feats = ["RSI", "Stoch_K", "ADX", "r1", "r5", "r10", "vol10", "mom10", "pma20", "pma50", "vratio", "rsi_d", "macd_h", "macd_hd", "bb_pos", "atr_p"]
        clean = df[feats + ["target"]].dropna()
        if len(clean) < 60: return None
        X = clean[feats]; y = clean["target"]; sp = int(len(X) * 0.8)
        Xtr, Xte = X.iloc[:sp], X.iloc[sp:]; ytr, yte = y.iloc[:sp], y.iloc[sp:]
        scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
        if HAS_LGBM: model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1)
        else: model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8)
        model.fit(Xtr_s, ytr); acc = accuracy_score(yte, model.predict(Xte_s))
        proba = model.predict_proba(scaler.transform(X.iloc[[-1]]))[0]
        top_f = sorted(zip(feats, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
        return {"hausse": proba[1], "baisse": proba[0], "acc": acc, "dir": "ACHAT" if proba[1] > 0.55 else "VENTE" if proba[0] > 0.55 else "NEUTRE", "conf": max(proba), "top_features": top_f}
    except: return None


# ══════════════════════════════════════════════════════════
# MULTI-TIMEFRAME
# ══════════════════════════════════════════════════════════

def analyser_mtf(ticker):
    result = {'4h': None, 'weekly': None, 'consensus': "NEUTRE"}
    try:
        data_4h = telecharger_4h(ticker)
        if not data_4h.empty and len(data_4h) >= 30:
            data_4h['RSI'] = calc_rsi(data_4h['Close']); data_4h['MACD'], data_4h['MACD_Signal'] = calc_macd(data_4h['Close'])
            d = data_4h.iloc[-1]; rsi_4h = float(d['RSI']) if not np.isnan(float(d['RSI'])) else 50
            macd_4h = float(d['MACD']); sig_4h = float(d['MACD_Signal'])
            sa = 0; sv = 0
            if rsi_4h < 35: sa += 1
            elif rsi_4h > 65: sv += 1
            if macd_4h > sig_4h: sa += 1
            else: sv += 1
            t = "ACHAT" if sa >= 2 else "VENTE" if sv >= 2 else "NEUTRE"
            result['4h'] = {'tendance': t, 'rsi': rsi_4h}
    except: pass
    try:
        data_w = telecharger_weekly(ticker)
        if not data_w.empty and len(data_w) >= 20:
            data_w['RSI'] = calc_rsi(data_w['Close']); data_w['MA_20'] = data_w['Close'].rolling(20).mean()
            d = data_w.iloc[-1]; rsi_w = float(d['RSI']) if not np.isnan(float(d['RSI'])) else 50
            px = float(d['Close']); ma20 = float(d['MA_20']) if not np.isnan(float(d['MA_20'])) else px
            if rsi_w < 40 and px > ma20: t = "ACHAT"
            elif rsi_w > 60 and px < ma20: t = "VENTE"
            elif px > ma20: t = "ACHAT"
            elif px < ma20: t = "VENTE"
            else: t = "NEUTRE"
            result['weekly'] = {'tendance': t, 'rsi': rsi_w}
    except: pass
    tendances = []
    if result['4h']: tendances.append(result['4h']['tendance'])
    if result['weekly']: tendances.append(result['weekly']['tendance'])
    if tendances.count("ACHAT") >= 2: result['consensus'] = "ACHAT"
    elif tendances.count("VENTE") >= 2: result['consensus'] = "VENTE"
    elif "ACHAT" in tendances and "VENTE" not in tendances: result['consensus'] = "ACHAT"
    elif "VENTE" in tendances and "ACHAT" not in tendances: result['consensus'] = "VENTE"
    return result


# ══════════════════════════════════════════════════════════
# ÉVALUATION (v7.1 — CONFIRMATION RETOURNEMENT)
# ══════════════════════════════════════════════════════════

def V(v):
    return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)


def evaluer(data, ticker, nom, macro_data, divergences=None):
    if len(data) < 2: return None
    last = data.iloc[-1]; prev = data.iloc[-2]; prix = V(last["Close"]); adx = V(last["ADX"])
    atr = V(last["ATR"]) if not np.isnan(V(last["ATR"])) else 0
    if np.isnan(adx) or adx < SEUIL_ADX:
        return {"action": "PLAT", "prix": prix, "adx": adx, "atr": atr, "score_achat": 0, "score_vente": 0, "details": [("ADX", "PLAT", f"ADX={round(adx, 1)} < {SEUIL_ADX}")], "sl_tp": None, "ml": None, "supports": [], "resistances": [], "divergences_txt": "", "mtf": None}
    sa = 0; sv = 0; det = []

    # RSI — CONFIRMATION RETOURNEMENT
    rsi = V(last["RSI"]); rsi_p = V(prev["RSI"])
    if rsi > 30 and rsi_p < 30: sa += POIDS["RSI"]; det.append(("RSI", "ACHAT", f"↗️ Sort de survendu ({round(rsi_p, 1)}→{round(rsi, 1)})"))
    elif rsi < 30 and rsi_p < 30: sa += POIDS["RSI"] * 0.3; det.append(("RSI", "ACHAT", f"⏳ Survendu ({round(rsi, 1)})"))
    elif rsi < 70 and rsi_p > 70: sv += POIDS["RSI"]; det.append(("RSI", "VENTE", f"↘️ Sort de suracheté ({round(rsi_p, 1)}→{round(rsi, 1)})"))
    elif rsi > 70 and rsi_p > 70: sv += POIDS["RSI"] * 0.3; det.append(("RSI", "VENTE", f"⏳ Suracheté ({round(rsi, 1)})"))
    else: det.append(("RSI", "—", f"{round(rsi, 1)}"))

    # MACD — CROISEMENT FRAIS
    m = V(last["MACD"]); ms = V(last["MACD_Signal"]); mp = V(prev["MACD"]); msp = V(prev["MACD_Signal"])
    if m > ms and mp <= msp: sa += POIDS["MACD"]; det.append(("MACD", "ACHAT", "↗️ Croisement haussier FRAIS"))
    elif m > ms and mp > msp: sa += POIDS["MACD"] * 0.6; det.append(("MACD", "ACHAT", "Bullish confirmé"))
    elif m < ms and mp >= msp: sv += POIDS["MACD"]; det.append(("MACD", "VENTE", "↘️ Croisement baissier FRAIS"))
    elif m < ms and mp < msp: sv += POIDS["MACD"] * 0.6; det.append(("MACD", "VENTE", "Bearish confirmé"))
    else: det.append(("MACD", "—", "Neutre"))

    # STOCHASTIQUE — CONFIRMATION RETOURNEMENT
    sk = V(last["Stoch_K"]); skp = V(prev["Stoch_K"])
    if sk > 20 and skp < 20: sa += POIDS["STOCH"]; det.append(("STOCH", "ACHAT", f"↗️ Sort de survendu ({round(skp, 1)}→{round(sk, 1)})"))
    elif sk < 20 and skp < 20: sa += POIDS["STOCH"] * 0.3; det.append(("STOCH", "ACHAT", f"⏳ Survendu ({round(sk, 1)})"))
    elif sk < 80 and skp > 80: sv += POIDS["STOCH"]; det.append(("STOCH", "VENTE", f"↘️ Sort de suracheté ({round(skp, 1)}→{round(sk, 1)})"))
    elif sk > 80 and skp > 80: sv += POIDS["STOCH"] * 0.3; det.append(("STOCH", "VENTE", f"⏳ Suracheté ({round(sk, 1)})"))
    else: det.append(("STOCH", "—", f"{round(sk, 1)}"))

    f6 = V(last["Fib_618"]); f3 = V(last["Fib_382"])
    if prix <= f6: sa += POIDS["FIBO"]; det.append(("FIBO", "ACHAT", "Sous 61.8%"))
    elif prix >= f3: sv += POIDS["FIBO"]; det.append(("FIBO", "VENTE", "Au-dessus 38.2%"))
    else: det.append(("FIBO", "—", "Entre niveaux"))

    ma = V(last["MA_200"])
    if not np.isnan(ma):
        if prix <= ma * 1.02: sa += POIDS["MA200"]; det.append(("MA200", "ACHAT", "Sous/proche MA200"))
        elif prix >= ma * 1.10: sv += POIDS["MA200"]; det.append(("MA200", "VENTE", "+10% au-dessus"))
        else: det.append(("MA200", "—", "Zone normale"))

    bbu = V(last["BB_Upper"]); bbl = V(last["BB_Lower"])
    if not np.isnan(bbu):
        if prix <= bbl: sa += POIDS["BOLLINGER"]; det.append(("BOLL", "ACHAT", "Bande basse"))
        elif prix >= bbu: sv += POIDS["BOLLINGER"]; det.append(("BOLL", "VENTE", "Bande haute"))
        else: det.append(("BOLL", "—", "Entre bandes"))

    vol = V(last["Volume"]); vm = V(last["Vol_Moy"])
    if not np.isnan(vm) and vm > 0:
        ratio_vol = vol / vm
        if ratio_vol >= 1.5:
            if sa > sv: sa += POIDS["VOLUME"]; det.append(("VOL", "ACHAT", f"{round(ratio_vol, 1)}x confirme"))
            elif sv > sa: sv += POIDS["VOLUME"]; det.append(("VOL", "VENTE", f"{round(ratio_vol, 1)}x confirme"))
        else: det.append(("VOL", "—", f"{round(ratio_vol, 1)}x"))

    div_tech = detecter_divergences_tech(data)
    if div_tech['rsi'] == "HAUSSIERE" or div_tech['macd'] == "HAUSSIERE": sa += POIDS["DIVERGENCE"]; det.append(("DIV_TECH", "ACHAT", "Divergence haussière"))
    elif div_tech['rsi'] == "BAISSIERE" or div_tech['macd'] == "BAISSIERE": sv += POIDS["DIVERGENCE"]; det.append(("DIV_TECH", "VENTE", "Divergence baissière"))
    else: det.append(("DIV_TECH", "—", "Pas de divergence"))

    try:
        tk = V(last["Ichi_TK"]); kj = V(last["Ichi_KJ"]); spa = V(last["Ichi_SpA"]); spb = V(last["Ichi_SpB"])
        if not np.isnan(spa) and not np.isnan(spb):
            nh = max(spa, spb); nb = min(spa, spb)
            if prix > nh and tk > kj: sa += POIDS["ICHIMOKU"]; det.append(("ICHI", "ACHAT", "Au-dessus nuage + TK>KJ"))
            elif prix < nb and tk < kj: sv += POIDS["ICHIMOKU"]; det.append(("ICHI", "VENTE", "Sous nuage + TK<KJ"))
            elif prix > nh: sa += POIDS["ICHIMOKU"] * 0.5; det.append(("ICHI", "ACHAT", "Au-dessus nuage"))
            elif prix < nb: sv += POIDS["ICHIMOKU"] * 0.5; det.append(("ICHI", "VENTE", "Sous nuage"))
            else: det.append(("ICHI", "—", "Dans le nuage"))
    except: det.append(("ICHI", "—", "N/A"))

    vwap = V(last["VWAP"])
    if not np.isnan(vwap):
        if prix < vwap * 0.99: sa += POIDS["VWAP"]; det.append(("VWAP", "ACHAT", "Sous VWAP"))
        elif prix > vwap * 1.01: sv += POIDS["VWAP"]; det.append(("VWAP", "VENTE", "Au-dessus VWAP"))
        else: det.append(("VWAP", "—", "Proche VWAP"))

    supports, resistances = detect_supports_resistances(data)
    sp_close = max([s for s in supports if s < prix], default=None)
    rs_close = min([r for r in resistances if r > prix], default=None)
    if sp_close and (prix - sp_close) / prix < 0.02: sa += POIDS["SUPPORTS_RES"]; det.append(("S/R", "ACHAT", f"Support {round(sp_close, 2)}"))
    elif rs_close and (rs_close - prix) / prix < 0.02: sv += POIDS["SUPPORTS_RES"]; det.append(("S/R", "VENTE", f"Résistance {round(rs_close, 2)}"))
    else: det.append(("S/R", "—", f"S:{round(sp_close, 2) if sp_close else '?'} R:{round(rs_close, 2) if rs_close else '?'}"))

    if ticker in CCXT_SYMBOLS:
        ob = fetch_orderbook(CCXT_SYMBOLS[ticker])
        if ob:
            imb = ob["imbalance"]
            if imb > 0.3: sa += POIDS["ORDER_FLOW"]; det.append(("FLOW", "ACHAT", f"+{round(imb * 100, 0)}%"))
            elif imb < -0.3: sv += POIDS["ORDER_FLOW"]; det.append(("FLOW", "VENTE", f"{round(imb * 100, 0)}%"))
            else: det.append(("FLOW", "—", f"{round(imb * 100, 0)}%"))
    else: det.append(("FLOW", "—", "N/A"))

    categorie = ACTIF_CATEGORIE.get(nom, "forex"); ms_val = calc_macro_score(macro_data, categorie)
    if ms_val >= 3: sa += POIDS["MACRO"]; det.append(("MACRO", "ACHAT", f"+{round(ms_val, 1)}/10"))
    elif ms_val <= -3: sv += POIDS["MACRO"]; det.append(("MACRO", "VENTE", f"{round(ms_val, 1)}/10"))
    elif ms_val >= 1: sa += POIDS["MACRO"] * 0.4; det.append(("MACRO", "ACHAT", f"Léger +{round(ms_val, 1)}"))
    elif ms_val <= -1: sv += POIDS["MACRO"] * 0.4; det.append(("MACRO", "VENTE", f"Léger {round(ms_val, 1)}"))
    else: det.append(("MACRO", "—", f"{round(ms_val, 1)}"))

    if macro_data.get("fear_greed"):
        fg = macro_data["fear_greed"].get("value", 50)
        if fg < 25: sa += POIDS["SENTIMENT"]; det.append(("SENT", "ACHAT", f"Extreme Fear ({fg})"))
        elif fg < 35: sa += POIDS["SENTIMENT"] * 0.5; det.append(("SENT", "ACHAT", f"Fear ({fg})"))
        elif fg > 75: sv += POIDS["SENTIMENT"]; det.append(("SENT", "VENTE", f"Extreme Greed ({fg})"))
        elif fg > 65: sv += POIDS["SENTIMENT"] * 0.5; det.append(("SENT", "VENTE", f"Greed ({fg})"))
        else: det.append(("SENT", "—", f"F&G={fg}"))

    ns, ns_detail, _ = get_news_score(ticker)
    if ns >= 4: sa += POIDS["NEWS_NLP"]; det.append(("NEWS", "ACHAT", f"+{round(ns, 1)} ({ns_detail})"))
    elif ns <= -4: sv += POIDS["NEWS_NLP"]; det.append(("NEWS", "VENTE", f"{round(ns, 1)} ({ns_detail})"))
    elif ns >= 2: sa += POIDS["NEWS_NLP"] * 0.4; det.append(("NEWS", "ACHAT", f"Léger + ({round(ns, 1)})"))
    elif ns <= -2: sv += POIDS["NEWS_NLP"] * 0.4; det.append(("NEWS", "VENTE", f"Léger - ({round(ns, 1)})"))
    else: det.append(("NEWS", "—", f"{round(ns, 1)}"))

    oc_score, _ = get_onchain(ticker)
    if oc_score >= 3: sa += POIDS["ONCHAIN"]; det.append(("CHAIN", "ACHAT", f"+{oc_score}"))
    elif oc_score <= -3: sv += POIDS["ONCHAIN"]; det.append(("CHAIN", "VENTE", f"{oc_score}"))
    elif oc_score >= 1: sa += POIDS["ONCHAIN"] * 0.4; det.append(("CHAIN", "ACHAT", "Léger +"))
    elif oc_score <= -1: sv += POIDS["ONCHAIN"] * 0.4; det.append(("CHAIN", "VENTE", "Léger -"))
    else: det.append(("CHAIN", "—", "N/A"))

    ml = ml_predict(ticker, data)
    if ml and ml["acc"] > 0.52:
        if ml["dir"] == "ACHAT" and ml["conf"] > 0.55: sa += POIDS["ML_PREDICTION"]; det.append(("🤖 ML", "ACHAT", f"{round(ml['hausse'] * 100, 0)}% (acc {round(ml['acc'] * 100, 0)}%)"))
        elif ml["dir"] == "VENTE" and ml["conf"] > 0.55: sv += POIDS["ML_PREDICTION"]; det.append(("🤖 ML", "VENTE", f"{round(ml['baisse'] * 100, 0)}% (acc {round(ml['acc'] * 100, 0)}%)"))
        else: det.append(("🤖 ML", "—", f"Confiance {round(ml['conf'] * 100, 0)}%"))
    else: det.append(("🤖 ML", "—", "Données insuffisantes"))

    sig_ob, msg_ob = indicateur_or_btc(ticker)
    if sig_ob == 1: sa += POIDS["OR_BTC"]; det.append(("OR/BTC", "ACHAT", msg_ob))
    elif sig_ob == -1: sv += POIDS["OR_BTC"]; det.append(("OR/BTC", "VENTE", msg_ob))
    else: det.append(("OR/BTC", "—", "Neutre"))

    div_txt = ""
    if divergences and ticker in ["GC=F", "SI=F", "PL=F"]:
        div_signal, div_detail = get_divergence_metal_impact(ticker, divergences)
        if div_signal == 1: sa += POIDS["DIV_METALS"]; det.append(("DIV_METALS", "ACHAT", div_detail)); div_txt = f"🟢 {div_detail}"
        elif div_signal == -1: sv += POIDS["DIV_METALS"]; det.append(("DIV_METALS", "VENTE", div_detail)); div_txt = f"🔴 {div_detail}"
        else: det.append(("DIV_METALS", "—", "Normal"))

    sl_tp = None
    if sa > sv:
        sl = prix - 1.5 * atr; tp = prix + 2.5 * atr
        if sp_close: sl = max(sl, sp_close * 0.998)
        if rs_close: tp = min(tp, rs_close * 0.998)
        risque = (prix - sl) / prix * 100; reward = (tp - prix) / prix * 100
        sl_tp = {"stop_loss": sl, "take_profit": tp, "risque_pct": risque, "reward_pct": reward, "ratio_rr": reward / risque if risque > 0 else 0, "atr": atr}
    elif sv > sa:
        sl = prix + 1.5 * atr; tp = prix - 2.5 * atr
        if rs_close: sl = min(sl, rs_close * 1.002)
        if sp_close: tp = max(tp, sp_close * 1.002)
        risque = (sl - prix) / prix * 100; reward = (prix - tp) / prix * 100
        sl_tp = {"stop_loss": sl, "take_profit": tp, "risque_pct": risque, "reward_pct": reward, "ratio_rr": reward / risque if risque > 0 else 0, "atr": atr}

    mtf = analyser_mtf(ticker)
    seuil = config.get("seuil", 8.0)
    if sa >= seuil and sa > sv: action = "ACHAT"
    elif sv >= seuil: action = "VENTE"
    else: action = "ATTENDRE"
    det.append(("ADX", "OK", f"ADX={round(adx, 1)} → Tendance"))
    if mtf:
        if mtf.get('4h'): det.append(("MTF 4H", mtf['4h']['tendance'], f"RSI: {round(mtf['4h']['rsi'], 0)}"))
        if mtf.get('weekly'): det.append(("MTF W", mtf['weekly']['tendance'], f"RSI: {round(mtf['weekly']['rsi'], 0)}"))
    return {"action": action, "prix": prix, "adx": adx, "atr": atr, "score_achat": sa, "score_vente": sv, "details": det, "sl_tp": sl_tp, "ml": ml, "supports": supports, "resistances": resistances, "divergences_txt": div_txt, "mtf": mtf}


# ══════════════════════════════════════════════════════════
# SCAN + BACKTEST + GRAPHIQUE + CHECK-LIST
# ══════════════════════════════════════════════════════════

def scan_actif(nom, ticker, macro_data, divergences):
    try:
        data = telecharger(ticker)
        if data is None or data.empty: return None
        # Vérifier que les colonnes essentielles existent
        colonnes_requises = ['Open', 'High', 'Low', 'Close', 'Volume']
        manquantes = [c for c in colonnes_requises if c not in data.columns]
        if manquantes:
            st.warning(f"⚠️ {nom}: colonnes manquantes {manquantes}")
            return None
        if len(data) < 50: return None
        data = calculer_indicateurs(data)
        if np.isnan(V(data.iloc[-1]["RSI"])): return None
  
        result = evaluer(data, ticker, nom, macro_data, divergences)
        if not result: return None
        tz = pytz.timezone("Europe/Zurich"); heure = datetime.now(tz).hour
        cat = ACTIF_CATEGORIE.get(nom, "forex"); h_info = HORAIRES.get(cat, HORAIRES["forex"])
        result.update({"nom": nom, "ticker": ticker, "score_max": SCORE_MAX, "data": data, "heure": heure, "heure_ok_buy": heure in h_info["buy"], "heure_ok_sell": heure in h_info["sell"], "heure_avoid": heure in h_info["avoid"], "buy_txt": h_info["buy_txt"], "sell_txt": h_info["sell_txt"]})
        return result
    except: return None


def lancer_scan(actifs, macro_data, divergences):
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(scan_actif, nom, ACTIFS[nom], macro_data, divergences): nom for nom in actifs}
        for f in as_completed(futures):
            r = f.result()
            if r: results.append(r)
    results.sort(key=lambda x: (0 if x["action"] in ["ACHAT", "VENTE"] else 1, -max(x["score_achat"], x["score_vente"])))
    return results


def graphique(data, nom, supports=None, resistances=None):
    df = data.tail(60)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25], subplot_titles=["Prix", "RSI", "MACD"])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Prix", line=dict(color="#667eea", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_200"], name="MA200", line=dict(color="#ffd700", dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB+", line=dict(color="rgba(255,100,100,0.5)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB-", line=dict(color="rgba(100,255,100,0.5)", width=1), fill="tonexty", fillcolor="rgba(100,100,255,0.05)"), row=1, col=1)
    if supports:
        for s in supports[:3]: fig.add_hline(y=s, line_dash="dot", line_color="cyan", row=1, col=1)
    if resistances:
        for r in resistances[:3]: fig.add_hline(y=r, line_dash="dot", line_color="red", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#a855f7", width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dash", line_color="cyan", row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#667eea", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#f5576c", width=1.5)), row=3, col=1)
    fig.update_layout(height=600, showlegend=True, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=50, r=20, t=40, b=20))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)"); fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def calculer_checklist(r, risque_pct):
    score = max(r.get("score_achat", 0), r.get("score_vente", 0)); adx = r.get("adx", 0)
    rr = r.get("sl_tp", {}).get("ratio_rr", 0) if r.get("sl_tp") else 0
    ml = r.get("ml"); action = r.get("action", "ATTENDRE")
    heure_ok = (r.get("heure_ok_buy") and action == "ACHAT") or (r.get("heure_ok_sell") and action == "VENTE")
    heure_avoid = r.get("heure_avoid", False)
    div_ok = True
    if r.get("ticker") in ["GC=F", "SI=F", "PL=F"]:
        for ind, sig, txt in r.get("details", []):
            if ind == "DIV_METALS" and sig == "VENTE" and action == "ACHAT": div_ok = False
            elif ind == "DIV_METALS" and sig == "ACHAT" and action == "VENTE": div_ok = False
    ml_ok = bool(ml and ml.get("acc", 0) > 0.52 and ml.get("dir") == action)
    risque_ok = r["sl_tp"].get("risque_pct", 0) <= risque_pct * 1.5 if r.get("sl_tp") else True
    mtf_ok = r.get("mtf", {}).get("consensus") == action if r.get("mtf") else False
    high_impact = check_high_impact_event(); no_event = len(high_impact) == 0
    return [("📊 Score ≥ 12", score >= 12), ("⚖️ R:R ≥ 1:1.5", rr >= 1.5), ("📈 ADX > 25", adx > 25), ("⏰ Bonne heure", heure_ok and not heure_avoid), ("🤖 ML confirme", ml_ok), ("🔀 Divergences OK", div_ok), ("📊 MTF aligné", mtf_ok), ("💰 Risque OK", risque_ok), ("📅 Pas d'événement", no_event)]


def afficher_checklist(r, risque_pct):
    criteres = calculer_checklist(r, risque_pct); score_ck = sum(1 for _, ok in criteres if ok)
    action = r.get("action", "?"); emoji = "🟢" if action == "ACHAT" else "🔴" if action == "VENTE" else "⏸️"
    st.markdown(f"### {emoji} {r.get('nom', '?')} — {action}")
    col_a, col_b = st.columns(2)
    for i, (label, ok) in enumerate(criteres):
        target = col_a if i < 5 else col_b
        with target: st.markdown(f"{'✅' if ok else '❌'} {label}")
    st.markdown("---")
    if score_ck >= 7: st.success(f"🟢 **GO** — {score_ck}/9 critères validés")
    elif score_ck >= 5: st.warning(f"🟡 **PRUDENCE** — {score_ck}/9")
    else: st.error(f"🔴 **STOP** — {score_ck}/9 (attends)")
    st.progress(score_ck / 9)

    # --- RÉSUMÉ DES CRITÈRES ---
    valides = [label for label, ok in criteres if ok]
    echoues = [label for label, ok in criteres if not ok]

    st.markdown("---")
    st.markdown("#### 📝 Résumé")
    if valides:
        st.markdown(f"**✅ Validés ({len(valides)})** : {' • '.join(valides)}")
    if echoues:
        st.markdown(f"**❌ Non validés ({len(echoues)})** : {' • '.join(echoues)}")

    return score_ck

    


def calculer_taille_position(capital, risque_pct, prix, stop_loss):
    risque_par_unite = abs(prix - stop_loss)
    if risque_par_unite == 0: return 0, 0
    montant_risque = capital * (risque_pct / 100); nb = montant_risque / risque_par_unite
    return nb, nb * prix


# ══════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ══════════════════════════════════════════════════════════

st.title("🧠 Trading Scanner v7.1")
st.caption(f"{'✅MT5' if HAS_MT5 else '⚠️MT5'} | {'✅ccxt' if HAS_CCXT else '⚠️ccxt'} | {'✅LGB' if HAS_LGBM else '⚠️GB'} | {'✅Trad' if HAS_TRANSLATOR else '⚠️Trad'} | Score max: {SCORE_MAX} | 📅 Calendrier + 🇫🇷 FR")

with st.sidebar:
    st.header("⚙️ Configuration")
    seuil = st.slider("Seuil d'alerte", 4.0, 20.0, config.get("seuil", 8.0), 0.5); config["seuil"] = seuil; save_config()
    st.divider()
    st.header("💰 Capital & Risque")
    capital = st.number_input("Capital (CHF)", 0, 1000000, config.get("capital", 1000), 100); config["capital"] = capital; save_config()
    risque_pct = st.slider("Risque/trade %", 0.5, 5.0, 2.0, 0.5)
    st.divider()
    st.header("📊 Actifs")
    defaut = config.get("actifs", ["🥇 Or (Gold)", "₿ Bitcoin", "💵 EUR/USD"])
    defaut = [a for a in defaut if a in ACTIFS] or ["🥇 Or (Gold)", "₿ Bitcoin", "💵 EUR/USD"]
    actifs_choisis = st.multiselect("Sélection", list(ACTIFS.keys()), default=defaut); config["actifs"] = actifs_choisis; save_config()
    st.divider()
    st.header("🔔 Alertes")
    alert_mode = st.radio("Mode", ["Aucune", "Email Bluewin", "Telegram"])
    email_addr = ""; email_pass = ""; tg_token = ""; tg_chat = ""
    if alert_mode == "Email Bluewin":
        email_addr = st.text_input("Email", placeholder="nom@bluewin.ch"); email_pass = st.text_input("Mot de passe", type="password")
    elif alert_mode == "Telegram":
        tg_token = st.text_input("Bot Token", value=config.get("tg_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=config.get("tg_chat", "")); config["tg_token"] = tg_token; config["tg_chat"] = tg_chat; save_config()
    st.divider()
    st.divider()
    st.header("🌍 Macro Live")
    macro_data, macro_details = fetch_macro()
    for d in macro_details[:6]: st.caption(d)

    # --- INTERPRÉTATION MACRO ---
    # Calcul du score moyen toutes catégories
    macro_scores = {}
    for cat in ["crypto", "commodities", "forex", "stocks"]:
        macro_scores[cat] = calc_macro_score(macro_data, cat)
    macro_moy = sum(macro_scores.values()) / len(macro_scores)

    if macro_moy >= 4:
        st.success(f"🟢 **FAVORABLE ACHAT** — Environnement macro porteur ({round(macro_moy, 1)}/10)")
    elif macro_moy >= 2:
        st.info(f"🟢 Léger avantage achat — Macro légèrement positive ({round(macro_moy, 1)}/10)")
    elif macro_moy <= -4:
        st.error(f"🔴 **PRUDENCE GÉNÉRALE** — Macro défavorable ({round(macro_moy, 1)}/10)")
    elif macro_moy <= -2:
        st.warning(f"🟡 Vigilance — Macro légèrement négative ({round(macro_moy, 1)}/10)")
    else:
        st.caption(f"⚖️ Neutre — Macro sans direction claire ({round(macro_moy, 1)}/10)")

    # Détail par catégorie
    st.caption(f"  Crypto: {round(macro_scores['crypto'], 1)} | Matières: {round(macro_scores['commodities'], 1)} | Forex: {round(macro_scores['forex'], 1)} | Actions: {round(macro_scores['stocks'], 1)}")

    # --- TABLEAU EXPLICATIF (dépliable) ---
    with st.expander("ℹ️ Comment lire le Macro Live ?"):
        st.markdown("""
| Situation | Message | Explication |
|---|---|---|
| Score ≥ 4 | 🟢 **FAVORABLE ACHAT** | Conditions macro très positives |
| Score 2 à 4 | 🟢 Léger avantage | Quelques signaux positifs |
| Score -2 à +2 | ⚖️ Neutre | Pas de direction claire |
| Score -2 à -4 | 🟡 Vigilance | Quelques signaux négatifs |
| Score ≤ -4 | 🔴 **PRUDENCE** | Conditions macro défavorables |

---

**Ce qui est analysé :**

| Donnée | 🟢 Positif (achat) | 🔴 Négatif (vente) |
|---|---|---|
| **Fear & Greed** | < 25 (Extreme Fear = opportunité) | > 75 (Extreme Greed = excès) |
| **VIX (volatilité)** | < 15 (marché calme) | > 30 (panique) |
| **Dollar (DXY)** | En baisse sous MA20 | En hausse au-dessus MA20 |
| **Taux 10 ans (TNX)** | < 3.5% ou en baisse | > 4.5% ou en hausse |
| **S&P 500 (SPY)** | Au-dessus MA20, +2%/5j | Sous MA20, -2%/5j |
| **Funding Rate BTC** | Négatif (shorts paient) | > 0.05% (longs paient trop) |

---

**Impact par catégorie :**

| Catégorie | Facteurs les + importants |
|---|---|
| **Crypto** | Fear&Greed (25%) + Funding (20%) + SPY (20%) |
| **Matières premières** | Dollar (30%) + Taux (25%) |
| **Forex** | Dollar (35%) + Taux (25%) + VIX (20%) |
| **Actions** | SPY (30%) + VIX (25%) + Taux (20%) |

---

**Logique :**
- Dollar baisse → Or/Crypto montent (relation inverse)
- VIX bas → marchés calmes → favorable aux achats
- Fear & Greed très bas → tout le monde a peur → souvent un bon moment pour acheter (contrarian)
- Taux élevés → mauvais pour actions/crypto (l'argent va vers les obligations)

**Poids dans le score final :** 2.5 points sur ~33 (influence forte).
        """)

    st.divider()
    st.header("⏰ Auto-Scan")
    auto_scan = st.toggle("Activer", False)
    auto_freq = st.selectbox("Fréquence", ["30 sec", "1 min", "5 min", "15 min", "30 min", "1 heure", "2 heures", "4 heures"], index=2)
    auto_sec = {"30 sec": 30, "1 min": 60, "5 min": 300, "15 min": 900, "30 min": 1800, "1 heure": 3600, "2 heures": 7200, "4 heures": 14400}[auto_freq]

c1, c2, c3 = st.columns([2, 1.2, 1.2])
with c1: btn_scan = st.button("🚀 SCANNER", type="primary", use_container_width=True)
with c2: btn_mt5 = st.button("🔌 MT5", use_container_width=True)
with c3: btn_refresh = st.button("🔄 Refresh", use_container_width=True)

if btn_refresh: st.cache_data.clear(); st.success("✅ Cache vidé")

# SCAN
if btn_scan:
    if not actifs_choisis: st.warning("Sélectionne au moins un actif")
    else:
        with st.spinner("🧠 Analyse v7.1..."):
            divergences = analyser_divergences_metals()
            resultats = lancer_scan(actifs_choisis, macro_data, divergences)
            st.session_state.derniers_resultats = resultats; st.session_state.scan_effectue = True; st.session_state.dernieres_divergences = divergences
            alertes = [r for r in resultats if r["action"] in ["ACHAT", "VENTE"]]
            for a in alertes:
                if alert_mode == "Telegram" and tg_token and tg_chat: envoyer_telegram(format_telegram(a), tg_token, tg_chat)
                elif alert_mode == "Email Bluewin" and email_addr and email_pass: envoyer_email(f"🚨 {'LONG' if a['action'] == 'ACHAT' else 'SHORT'} — {a['nom']}", f"Prix: {round(a['prix'], 2)}", email_addr, email_pass)
                st.session_state.historique_signaux.append({"time": datetime.now(pytz.timezone("Europe/Zurich")).strftime("%H:%M"), "nom": a["nom"], "action": a["action"], "score": round(max(a["score_achat"], a["score_vente"]), 1)})
        st.rerun()

# RÉSULTATS
if st.session_state.scan_effectue and st.session_state.derniers_resultats:
    resultats = st.session_state.derniers_resultats
    st.markdown("---"); st.subheader("📋 Résultats")
    cols = st.columns(min(len(resultats), 4))
    for i, r in enumerate(resultats):
        with cols[i % len(cols)]:
            if r["action"] == "ACHAT": st.metric(r["nom"], f"{round(r['prix'], 2)}", f"🟢 LONG ({round(r['score_achat'], 1)})")
            elif r["action"] == "VENTE": st.metric(r["nom"], f"{round(r['prix'], 2)}", f"🔴 SHORT ({round(r['score_vente'], 1)})", delta_color="inverse")
            elif r["action"] == "PLAT": st.metric(r["nom"], f"{round(r['prix'], 2)}", "😴 Plat", delta_color="off")
            else: st.metric(r["nom"], f"{round(r['prix'], 2)}", f"⏸️ ({round(max(r['score_achat'], r['score_vente']), 1)})", delta_color="off")
    st.markdown("---"); st.subheader("✅ Check-list (9 critères)")
    noms = [r["nom"] for r in resultats]; actif_ck = st.selectbox("Actif", noms, key="sel_ck")
    r_c = next((r for r in resultats if r["nom"] == actif_ck), None)
    if r_c:
        with st.container(border=True): afficher_checklist(r_c, risque_pct)
    st.divider(); st.subheader("📊 Détails")
    for idx, r in enumerate(resultats):
        ic = "🟢" if r["action"] == "ACHAT" else "🔴" if r["action"] == "VENTE" else "😴" if r["action"] == "PLAT" else "🟡"
        lb = "LONG" if r["action"] == "ACHAT" else "SHORT" if r["action"] == "VENTE" else r["action"]
        with st.expander(f"{ic} {r['nom']} — {lb}", expanded=(r["action"] in ["ACHAT", "VENTE"])):
            if r.get("sl_tp") and r["action"] in ["ACHAT", "VENTE"]:
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🛑 SL", f"{round(r['sl_tp']['stop_loss'], 2)}"); sc2.metric("🎯 TP", f"{round(r['sl_tp']['take_profit'], 2)}"); sc3.metric("R:R", f"1:{round(r['sl_tp']['ratio_rr'], 1)}")
            fig = graphique(r["data"], r["nom"], r.get("supports"), r.get("resistances"))
            st.plotly_chart(fig, use_container_width=True, key=f"g_{idx}")
            for ind, sig, txt in r["details"]:
                if sig == "ACHAT": st.write(f"✅ **{ind}** — {txt}")
                elif sig == "VENTE": st.write(f"❌ **{ind}** — {txt}")
                else: st.write(f"⏸️ **{ind}** — {txt}")
     # ══════════════════════════════════════════════════════════
    # 🧠 FENÊTRE INTERPRÉTATION & ANALYSE (dépliable)
    # ══════════════════════════════════════════════════════════
    st.divider()
    with st.expander("🧠 Interprétation & Analyse détaillée", expanded=False):

        actif_interp = st.selectbox("Choisir un actif", noms, key="sel_interp")
        r_interp = next((r for r in resultats if r["nom"] == actif_interp), None)

        if r_interp:
            action = r_interp["action"]
            score_a = r_interp["score_achat"]
            score_v = r_interp["score_vente"]
            score_dominant = max(score_a, score_v)
            pct_score = round(score_dominant / SCORE_MAX * 100, 0)

            # --- RÉSUMÉ RAPIDE ---
            if action == "ACHAT":
                st.success(f"🟢 **Signal ACHAT** — Score {round(score_a, 1)}/{round(SCORE_MAX, 1)} ({pct_score}%)")
            elif action == "VENTE":
                st.error(f"🔴 **Signal VENTE** — Score {round(score_v, 1)}/{round(SCORE_MAX, 1)} ({pct_score}%)")
            elif action == "PLAT":
                st.warning(f"😴 **Marché PLAT** — ADX trop faible ({round(r_interp['adx'], 1)}), pas de tendance")
            else:
                st.info(f"⏸️ **ATTENDRE** — Score {round(score_dominant, 1)}/{round(SCORE_MAX, 1)} ({pct_score}%) — Seuil requis : {config.get('seuil', 8.0)}")

            st.markdown("---")

            # --- SCORES VISUELS ---
            st.markdown("#### 📊 Scores")
            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                st.caption(f"🟢 Achat : {round(score_a, 1)} / {round(SCORE_MAX, 1)}")
                st.progress(min(score_a / SCORE_MAX, 1.0))
            with col_bar2:
                st.caption(f"🔴 Vente : {round(score_v, 1)} / {round(SCORE_MAX, 1)}")
                st.progress(min(score_v / SCORE_MAX, 1.0))

            st.caption(f"Seuil pour déclencher un signal : **{config.get('seuil', 8.0)}** — Plus le score est élevé, plus les indicateurs convergent.")

            st.markdown("---")

            # --- SIGNAUX QUI CONTRIBUENT ---
            st.markdown("#### ✅ Signaux actifs")
            signaux_achat = [(ind, txt) for ind, sig, txt in r_interp["details"] if sig == "ACHAT"]
            signaux_vente = [(ind, txt) for ind, sig, txt in r_interp["details"] if sig == "VENTE"]
            signaux_neutres = [(ind, txt) for ind, sig, txt in r_interp["details"] if sig not in ["ACHAT", "VENTE"]]

            if signaux_achat:
                st.markdown("**🟢 Haussiers :**")
                for ind, txt in signaux_achat:
                    st.markdown(f"  - **{ind}** : {txt}")

            if signaux_vente:
                st.markdown("**🔴 Baissiers :**")
                for ind, txt in signaux_vente:
                    st.markdown(f"  - **{ind}** : {txt}")

            if signaux_neutres:
                st.markdown(f"**⏸️ Neutres ({len(signaux_neutres)}) :**")
                for ind, txt in signaux_neutres:
                    st.caption(f"{ind} : {txt}")

            st.markdown("---")

            # --- ANALYSE CONTEXTUELLE ---
            st.markdown("#### 🔍 Contexte")

            # ADX (force tendance)
            adx_val = r_interp["adx"]
            if adx_val > 40:
                st.markdown(f"📈 **Tendance très forte** (ADX {round(adx_val, 1)}) → Mouvement puissant")
            elif adx_val > 30:
                st.markdown(f"📈 **Tendance correcte** (ADX {round(adx_val, 1)}) → Signal fiable")
            elif adx_val > 25:
                st.markdown(f"📊 **Tendance faible** (ADX {round(adx_val, 1)}) → Prudence")
            else:
                st.markdown(f"😴 **Pas de tendance** (ADX {round(adx_val, 1)}) → Éviter")

            # Ratio R:R
            if r_interp.get("sl_tp"):
                rr = r_interp["sl_tp"]["ratio_rr"]
                risque = r_interp["sl_tp"]["risque_pct"]
                reward = r_interp["sl_tp"]["reward_pct"]
                if rr >= 2:
                    st.markdown(f"⚖️ **Excellent R:R** 1:{round(rr, 1)} — Risque {round(risque, 1)}% / Gain {round(reward, 1)}%")
                elif rr >= 1.5:
                    st.markdown(f"⚖️ **Bon R:R** 1:{round(rr, 1)} — Acceptable")
                elif rr >= 1:
                    st.markdown(f"⚖️ **R:R moyen** 1:{round(rr, 1)} — Limite")
                else:
                    st.markdown(f"⚠️ **R:R insuffisant** 1:{round(rr, 1)} — Trop risqué")

            # ML
            ml = r_interp.get("ml")
            if ml and ml.get("acc", 0) > 0.52:
                st.markdown(f"🤖 **ML** : prédit {'hausse' if ml['hausse'] > 0.5 else 'baisse'} à {round(max(ml['hausse'], ml['baisse']) * 100, 0)}% (fiabilité {round(ml['acc'] * 100, 0)}%)")
            else:
                st.markdown("🤖 **ML** : confiance insuffisante")

            # MTF
            mtf = r_interp.get("mtf")
            if mtf:
                details_mtf = []
                if mtf.get("4h"):
                    details_mtf.append(f"4H → {mtf['4h']['tendance']}")
                if mtf.get("weekly"):
                    details_mtf.append(f"Hebdo → {mtf['weekly']['tendance']}")
                aligned = mtf.get("consensus") == action
                st.markdown(f"🕐 **Multi-timeframe** : {'✅ Aligné' if aligned else '⚠️ Non aligné'} — {' | '.join(details_mtf)}")

            # Horaire
            heure_ok = (r_interp.get("heure_ok_buy") and action == "ACHAT") or (r_interp.get("heure_ok_sell") and action == "VENTE")
            heure_avoid = r_interp.get("heure_avoid", False)
            if heure_avoid:
                st.markdown("⏰ **Mauvais créneau** ❌")
            elif heure_ok:
                st.markdown("⏰ **Bon créneau** ✅")
            else:
                st.markdown(f"⏰ Créneau neutre (idéal achat : {r_interp.get('buy_txt', '?')} | vente : {r_interp.get('sell_txt', '?')})")

            st.markdown("---")

            # --- CONCLUSION ---
            st.markdown("#### 💡 Conclusion")
            nb_achat = len(signaux_achat)
            nb_vente = len(signaux_vente)
            nb_total = nb_achat + nb_vente + len(signaux_neutres)

            if action == "ACHAT":
                st.markdown(f"✅ **{nb_achat}/{nb_total} indicateurs** pointent vers un achat. Score {round(score_a, 1)} > seuil {config.get('seuil', 8.0)}.")
                if not heure_ok or heure_avoid:
                    st.caption("⚠️ Le timing horaire n'est pas idéal.")
            elif action == "VENTE":
                st.markdown(f"❌ **{nb_vente}/{nb_total} indicateurs** pointent vers une vente. Score {round(score_v, 1)} > seuil {config.get('seuil', 8.0)}.")
                if not heure_ok or heure_avoid:
                    st.caption("⚠️ Le timing horaire n'est pas idéal.")
            elif action == "PLAT":
                st.markdown("😴 Pas de direction claire (ADX < 25). **Reste en dehors.**")
            else:
                st.markdown(f"⏸️ Score {round(score_dominant, 1)} < seuil {config.get('seuil', 8.0)}. **{nb_achat} haussiers** vs **{nb_vente} baissiers** → Pas assez de convergence. **Attends.**")

            # --- TAILLE POSITION ---
            if action in ["ACHAT", "VENTE"] and r_interp.get("sl_tp"):
                st.markdown("---")
                st.markdown("#### 💰 Position suggérée")
                nb_units, montant = calculer_taille_position(capital, risque_pct, r_interp["prix"], r_interp["sl_tp"]["stop_loss"])
                st.markdown(f"Capital **{capital} CHF** | Risque **{risque_pct}%** = {round(capital * risque_pct / 100, 2)} CHF")
                st.markdown(f"→ **{round(nb_units, 4)} unités** (~{round(montant, 2)} CHF)")
    # ══════════════════════════════════════════════════════════
    # 🧠 FENÊTRE Explication critères  ANALYSE (dépliable)
    # ══════════════════════════════════════════════════════════
    st.divider()
    with st.expander("🧠 Explication critères", expanded=False):

        critere_choisi = st.selectbox("Choisir un critère à expliquer", [
            "RSI", "Stochastique", "MACD", "Fibonacci", "Moyenne Mobile 200", "ADX", "Convergence"
        ], key="sel_explication")

        # Explication simple de chaque critère
        explications = {
            "RSI": """
**RSI (Relative Strength Index)** – Mesure si un actif est suracheté ou survendu.
- 📉 **RSI < 30** → Survendu → **Signal d'achat**
- 📈 **RSI > 70** → Suracheté → **Signal de vente**
- Entre 30 et 70 → Neutre

*Imagine un élastique : plus il est étiré, plus il reviendra.*
            """,
            "Stochastique": """
**Stochastique** – Compare le prix actuel au range des derniers jours.
- 📉 **Stoch < 20** → Bas du range → **Signal d'achat**
- 📈 **Stoch > 80** → Haut du range → **Signal de vente**

*Thermomètre : en bas = froid (achat), en haut = chaud (vente).*
            """,
            "MACD": """
**MACD** – Détecte les changements de tendance.
- 📈 **MACD > ligne signal** → Tendance haussière → **Signal d'achat**
- 📉 **MACD < ligne signal** → Tendance baissière → **Signal de vente**

*Quand deux moyennes se croisent = changement de direction.*
            """,
            "Fibonacci": """
**Retracement de Fibonacci** – Niveaux naturels de support/résistance.
- 📉 **Prix sous 61.8%** → Zone de support → **Signal d'achat**
- 📈 **Prix au-dessus de 38.2%** → Zone de résistance → **Signal de vente**

*Des "paliers" où le prix a tendance à rebondir ou bloquer.*
            """,
            "Moyenne Mobile 200": """
**MA200 (Moyenne Mobile 200 jours)** – La tendance de fond sur ~1 an.
- 📉 **Prix sous la MA200** → Tendance baissière → **Signal d'achat**
- 📈 **Prix > 10% au-dessus** → Trop éloigné → **Signal de vente**

*C'est la "température moyenne". S'en éloigner trop = retour probable.*
            """,
            "ADX": """
**ADX (Average Directional Index)** – Mesure la force de la tendance.
- **ADX < 20** → Pas de tendance → Marché PLAT (on n'entre pas)
- **ADX > 20** → Tendance en cours → Les signaux sont fiables

*C'est le "volume sonore" de la tendance. Trop bas = bruit, pas de signal.*
            """,
            "Convergence": """
**Convergence (la logique du scanner)** – On ne suit pas UN seul indicateur.
- Chaque critère donne un score pondéré
- On additionne tous les scores → **Score total**
- Si le score dépasse le **seuil** → Alerte déclenchée

⚡ **Plus il y a de critères qui convergent, plus le signal est fiable.**

*C'est comme demander l'avis à 5 experts : si 3+ sont d'accord, on agit.*
            """
        }

        st.markdown(explications[critere_choisi])

              
elif not st.session_state.scan_effectue:
    # Auto-scan au premier chargement
    if actifs_choisis:
        with st.spinner("🧠 Scan automatique au démarrage..."):
            divergences = analyser_divergences_metals()
            resultats = lancer_scan(actifs_choisis, macro_data, divergences)
            st.session_state.derniers_resultats = resultats
            st.session_state.scan_effectue = True
            st.session_state.dernieres_divergences = divergences
        st.rerun()
    else:
        st.info("👆 Sélectionne des actifs dans la barre latérale")


# CALENDRIER ÉCONOMIQUE
st.divider(); st.header("📅 Calendrier Économique USA")
high_impact = check_high_impact_event()
if high_impact:
    st.error(f"🚨 **⛔ NE PAS TRADER** — {len(high_impact)} événement(s) majeur(s) à venir !")
    st.warning("Les annonces macro créent des mouvements violents et imprévisibles. Attends la publication.")
    for e in high_impact[:5]:
        date_hi = e.get('date', '?')
        st.markdown(f"{e['importance']} **{traduire(e['title'][:80])}** — 🕐 {date_hi}")
else:
    st.success("✅ **SAFE TO TRADE** — Aucun événement majeur imminent")
    st.caption("Pas de Fed, CPI, NFP ou FOMC dans les prochaines heures. Tu peux trader sereinement.")

events = get_economic_calendar()
nb_rouge = sum(1 for e in events if e.get("importance") == "🔴")
nb_orange = sum(1 for e in events if e.get("importance") == "🟠")

if events:
    st.caption(f"📊 {len(events)} événements détectés — {nb_rouge} 🔴 critiques — {nb_orange} 🟠 importants")
    with st.expander(f"📋 Détails ({len(events)} événements)"):
        events_sorted = sorted(events, key=lambda x: x.get('date', ''), reverse=True)
        for e in events_sorted[:15]:
            imp = e.get("importance", "⚪"); sc = e.get("score", 0)
            emoji = "🟢" if sc > 0.1 else "🔴" if sc < -0.1 else "⚪"
            upcoming = " 🔜" if e.get("upcoming") else ""
            date_e = e.get('date', '?')
            titre_e = traduire(e['title'][:70])
            if imp == "🔴":
                st.markdown(f"**{imp} {emoji} {titre_e}{upcoming} — 🕐 {date_e}**")
            elif imp == "🟠":
                st.markdown(f"{imp} {emoji} {titre_e}{upcoming} — 🕐 {date_e}")
            else:
                st.caption(f"{imp} {emoji} {titre_e}{upcoming} — 🕐 {date_e}")

# --- TABLEAU EXPLICATIF CALENDRIER ---
with st.expander("ℹ️ Comment lire le Calendrier Économique ?"):
    st.markdown("""
| Couleur | Événement | Impact sur le trading |
|---|---|---|
| 🔴 **Critique** | Fed, FOMC, Powell, Rate Decision | ⛔ **NE PAS TRADER** — Mouvement violent dans les 2 sens |
| 🟠 **Important** | CPI, NFP, Inflation, Employment | ⚠️ **Prudence** — Volatilité forte possible |
| 🟡 **Modéré** | GDP, PMI, Retail Sales | 🟡 Surveiller — Impact modéré |

---

**Règles simples :**
- 🔴 à venir → **Ferme tes positions ou ne rentre pas**
- 🟠 à venir → **Réduis ta taille de position**
- Rien de majeur → **✅ Safe to trade**

**Pourquoi c'est important :**
- Un CPI inattendu peut faire bouger l'Or de 2% en 5 minutes
- Une décision de la Fed peut retourner tout le marché
- Le scanner ne peut PAS prédire ces annonces

**Timing :**
- La plupart des annonces US sortent à **14h30** ou **20h00** (heure suisse)
- Éviter de trader 30 min avant ET 30 min après l'annonce

**Poids dans la check-list :** 1 critère sur 9 ("Pas d'événement").
    """)




# NEWS EN FRANÇAIS
st.divider(); st.header("📰 Sentiment News 🇫🇷")

# Scores news
ns, nd, nh = get_news_score("BTC-USD")
ns2, nd2, nh2 = get_news_score("GC=F")
ns3, nd3, nh3 = get_news_score("^GSPC")
news_moy = (ns + ns2 + ns3) / 3

# --- INTERPRÉTATION GLOBALE ---
if news_moy >= 4:
    st.success(f"🟢 **SENTIMENT TRÈS POSITIF** — Les news soutiennent l'achat (moy. {round(news_moy, 1)}/10)")
elif news_moy >= 2:
    st.info(f"🟢 Sentiment légèrement positif — News plutôt favorables (moy. {round(news_moy, 1)}/10)")
elif news_moy <= -4:
    st.error(f"🔴 **SENTIMENT TRÈS NÉGATIF** — Les news poussent à la prudence (moy. {round(news_moy, 1)}/10)")
elif news_moy <= -2:
    st.warning(f"🟡 Sentiment légèrement négatif — Quelques mauvaises news (moy. {round(news_moy, 1)}/10)")
else:
    st.caption(f"⚖️ Sentiment neutre — Pas de direction claire dans les news (moy. {round(news_moy, 1)}/10)")

cn1, cn2, cn3 = st.columns(3)
with cn1:
    st.subheader("Bitcoin"); st.metric("Score", f"{round(ns, 1)}/10"); st.caption(nd)
    for h in nh[:3]:
        emoji_h = '🟢' if h['score'] > 0.1 else '🔴' if h['score'] < -0.1 else '⚪'
        titre_h = traduire(h['title'][:60])
        date_h = h.get('date', '?')
        is_recent = (time.time() - h.get('date_raw', 0)) < 86400
        if abs(h['score']) > 0.3 and is_recent:
            st.markdown(f"**{emoji_h} {titre_h}** — 🕐 {date_h}")
        else:
            st.caption(f"{emoji_h} {titre_h} — 🕐 {date_h}")

with cn2:
    st.subheader("Or 🥇"); st.metric("Score", f"{round(ns2, 1)}/10"); st.caption(nd2)
    for h in nh2[:3]:
        emoji_h = '🟢' if h['score'] > 0.1 else '🔴' if h['score'] < -0.1 else '⚪'
        titre_h = traduire(h['title'][:60])
        date_h = h.get('date', '?')
        is_recent = (time.time() - h.get('date_raw', 0)) < 86400
        if abs(h['score']) > 0.3 and is_recent:
            st.markdown(f"**{emoji_h} {titre_h}** — 🕐 {date_h}")
        else:
            st.caption(f"{emoji_h} {titre_h} — 🕐 {date_h}")
with cn3:
    st.subheader("S&P 500"); st.metric("Score", f"{round(ns3, 1)}/10"); st.caption(nd3)
    for h in nh3[:3]:
        emoji_h = '🟢' if h['score'] > 0.1 else '🔴' if h['score'] < -0.1 else '⚪'
        titre_h = traduire(h['title'][:60])
        date_h = h.get('date', '?')
        is_recent = (time.time() - h.get('date_raw', 0)) < 86400
        if abs(h['score']) > 0.3 and is_recent:
            st.markdown(f"**{emoji_h} {titre_h}** — 🕐 {date_h}")
        else:
            st.caption(f"{emoji_h} {titre_h} — 🕐 {date_h}")

# --- TABLEAU EXPLICATIF NEWS ---
with st.expander("ℹ️ Comment lire le Sentiment News ?"):
    st.markdown("""
| Score | Message | Signification |
|---|---|---|
| ≥ 4 | 🟢 **TRÈS POSITIF** | Majorité de news bullish → favorable achat |
| 2 à 4 | 🟢 Légèrement positif | Plus de bonnes que de mauvaises news |
| -2 à +2 | ⚖️ Neutre | Pas de direction claire |
| -2 à -4 | 🟡 Légèrement négatif | Plus de mauvaises news |
| ≤ -4 | 🔴 **TRÈS NÉGATIF** | Majorité de news bearish → prudence |

---

**Comment c'est calculé :**
- Analyse des 25 derniers articles Google News + Kitco (pour l'Or)
- Chaque titre est évalué par un algorithme NLP (VADER)
- Mots bullish détectés : rally, surge, breakout, approval, rate cut...
- Mots bearish détectés : crash, plunge, selloff, ban, recession...
- Les articles récents ont plus de poids que les anciens

**Poids dans le score final :** 2.0 points sur ~33 (influence modérée).

**⚠️ Limites :**
- Basé sur les titres seulement (pas le contenu complet)
- News en anglais traduites automatiquement
- Peut avoir du retard si Google News met à jour lentement
    """)

# ON-CHAIN
st.divider(); st.header("⛓️ On-Chain Bitcoin")
oc_s, oc_d = get_onchain("BTC-USD"); st.metric("Score", f"{oc_s}/10")
for d in oc_d: st.write(d)

# --- INTERPRÉTATION ---
if oc_s >= 5:
    st.success(f"🟢 **FAVORABLE ACHAT BTC** — Réseau sain, fondamentaux solides (score {oc_s}/10)")
elif oc_s >= 3:
    st.success(f"🟢 **Léger avantage achat BTC** — Signaux on-chain positifs (score {oc_s}/10)")
elif oc_s <= -5:
    st.error(f"🔴 **PRUDENCE BTC** — Réseau en surchauffe, correction possible (score {oc_s}/10)")
elif oc_s <= -3:
    st.warning(f"🟡 **Vigilance BTC** — Quelques signaux négatifs on-chain (score {oc_s}/10)")
else:
    st.caption(f"⚖️ Neutre — Pas de signal on-chain clair (score {oc_s}/10)")

# --- TABLEAU EXPLICATIF (dépliable) ---
with st.expander("ℹ️ Comment lire les données On-Chain ?"):
    st.markdown("""
| Situation | Message | Explication |
|---|---|---|
| Score ≥ 5 | 🟢 **FAVORABLE ACHAT** | Hashrate monte, fees bas, peu de longs → réseau sain |
| Score 3 à 5 | 🟢 Léger avantage achat | Quelques signaux positifs |
| Score -3 à +3 | ⚖️ Neutre | Rien de remarquable |
| Score -3 à -5 | 🟡 Vigilance | Quelques signaux négatifs |
| Score ≤ -5 | 🔴 **PRUDENCE** | Fees élevés, trop de longs, TVL baisse → surchauffe |

---

**Ce qui est analysé :**

| Donnée | 🟢 Positif | 🔴 Négatif |
|---|---|---|
| **Fees (frais réseau)** | < 10 sat/vB (calme) | > 100 sat/vB (congestion) |
| **Mempool (TX en attente)** | < 10'000 TX | > 100'000 TX |
| **Hashrate (puissance mineurs)** | En hausse +5% | En baisse -5% |
| **Long/Short Ratio** | < 0.8 (shorts dominent) | > 2.0 (trop de longs) |
| **TVL DeFi** | En hausse +5% | En baisse -5% |

---

**Logique :** Le prix montre la surface. Le on-chain montre les **fondations**.
- Fees bas + Hashrate monte = mineurs confiants, réseau pas saturé → bon moment
- Fees explosent + tout le monde est long = euphorie → sommet probable

**Poids dans le score :** 2.5 points sur ~33 (influence moyenne-forte).

**Limites :** Fonctionne uniquement pour BTC/ETH/SOL. Dépend des API externes.
    """)


# OR vs BTC
st.divider(); st.header("🔗 Or vs Bitcoin (7j)")
div_ob = get_divergence_or_btc()
if div_ob:
    do1, do2, do3 = st.columns(3)
    do1.metric("Or", f"{round(div_ob['var_or'], 1)}%")
    do2.metric("BTC", f"{round(div_ob['var_btc'], 1)}%")
    do3.metric("Écart", f"{round(div_ob['ecart'], 1)}%")

    # --- INTERPRÉTATION ---
    ecart = div_ob["ecart"]
    if ecart < -5:
        st.success(f"🟢 **FAVORABLE ACHAT OR** — BTC surperforme de {abs(round(ecart, 1))}% → L'Or devrait rattraper")
        st.error(f"🔴 **PRUDENCE CRYPTO** — BTC en excès, correction possible")
    elif ecart > 5:
        st.success(f"🟢 **FAVORABLE ACHAT CRYPTO** — Or surperforme de {round(ecart, 1)}% → Crypto devrait rattraper")
        st.error(f"🔴 **PRUDENCE OR** — Or en excès, correction possible")
    elif ecart < -3:
        st.info(f"🟡 **Léger avantage Or** — BTC avance un peu plus ({abs(round(ecart, 1))}%), surveille si ça s'accentue")
    elif ecart > 3:
        st.info(f"🟡 **Léger avantage Crypto** — Or avance un peu plus ({round(ecart, 1)}%), surveille si ça s'accentue")
    else:
        st.caption(f"⚖️ Équilibre — Écart {round(ecart, 1)}% (seuil ±5%). Pas de signal.")
    # --- TABLEAU EXPLICATIF (dépliable) ---
    with st.expander("ℹ️ Comment lire l'écart Or vs Bitcoin ?"):
        st.markdown("""
| Situation | Message |
|---|---|
| BTC +8%, Or +1% (écart < -5%) | 🟢 **FAVORABLE ACHAT OR** + 🔴 **PRUDENCE CRYPTO** |
| Or +7%, BTC +1% (écart > +5%) | 🟢 **FAVORABLE ACHAT CRYPTO** + 🔴 **PRUDENCE OR** |
| Écart entre ±3% et ±5% | 🟡 Surveille, ça pourrait s'accentuer |
| Écart < ±3% | ⚖️ Équilibre, pas de signal |

**Logique :** Or et Bitcoin sont des valeurs refuge. Quand l'un prend trop d'avance (+5%), l'autre a tendance à rattraper.

**Poids dans le score :** 1.5 points (confirmation, pas déclencheur principal).
        """)


# HISTORIQUE
if st.session_state.historique_signaux:
    st.divider(); st.header("📜 Signaux (session)")
    for s in reversed(st.session_state.historique_signaux[-10:]): st.caption(f"{s['time']} | {'🟢' if s['action'] == 'ACHAT' else '🔴'} {s['nom']} — {s['score']}")

# FOOTER
st.markdown("---")
st.caption(f"🧠 Trading Scanner v7.1 | ⚠️ Pas un conseil financier | {len(ACTIFS)} actifs | Score max: {SCORE_MAX} | 📅 Calendrier + 🇫🇷 News FR")

# AUTO-SCAN
if auto_scan and actifs_choisis:
    st.subheader(f"🔄 Auto-scan — {auto_freq}"); ph = st.empty(); ct = st.empty()
    while True:
        with ph.container():
            st.caption(f"⏰ {datetime.now(pytz.timezone('Europe/Zurich')).strftime('%H:%M:%S')}")
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
            m_r, s_r = divmod(rem, 60); ct.caption(f"⏳ {m_r:02d}:{s_r:02d}"); time.sleep(1)
        st.rerun()
