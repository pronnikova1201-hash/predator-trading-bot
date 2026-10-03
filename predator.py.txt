import os
import time
import datetime
import math
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from pybit.unified_trading import HTTP
import fcntl
import sys
from scipy.signal import argrelextrema
import logging
import sqlite3
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict
from functools import wraps

# --- ПАТЧ: МАШИННОЕ ОБУЧЕНИЕ ДЛЯ УРОВНЕЙ ---
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    print("ОШИБКА: Не установлена библиотека scikit-learn. Выполните: pip install scikit-learn")
    sys.exit(1)
# -------------------------------------------

# Настраиваем логирование
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# ==========================================
# 🔒 ЗАЩИТА ОТ ДВОЙНОГО ЗАПУСКА
# ==========================================
lock_file = open('predator.lock', 'w')
try:
    fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except IOError:
    sys.exit(0)

# ==========================================
# ⚙️ НАСТРОЙКИ И CONFIG
# ==========================================
CONFIG = {
    # === RISK_REDUCTION_PATCH_V9 ===
    # === COST_GUARD_AND_R_TRAIL_PATCH_V23 ===
    "LEVERAGE": 10,
    # Имя историческое (v28), фильтр живой с v29: мин. фитиль ложного пробоя в ATR
    "SHADOW_MIN_TAIL_ATR_MULT": 0.3,
    # === STRUCTURAL_EXIT_GRACE_PATCH_V16 ===
    "MIN_TIME_IN_TRADE_SECONDS": 600,
    "MIN_STRUCTURAL_EXIT_PCT": 0.006,
    # --- РИСК-МЕНЕДЖМЕНТ (FIX-5) ---
    "RISK_PER_TRADE_PCT": 1.0,     # риск 1% баланса на сделку, от ширины стопа
    "MAX_MARGIN_USAGE": 0.5,       # потолок: не более 50% доступной маржи
    "MIN_NOTIONAL_USDT": 5.0,      # ниже — Bybit отклонит ордер

    # --- COST GUARD (FIX-3) ---
    "COST_ROUND_TRIP_BPS": 24.0,   # тейкер 5.5*2 + спред 2 + буфер 3 (обновлено v23)
    "MIN_EDGE_COST_MULT": 3.0,     # стоп >= 3x издержек
    "MIN_STOP_DISTANCE_BPS": 70.0, # абсолютный пол стопа: 0.70% (обновлено v23)
    "MIN_STOP_ATR_MULT": 1.5,      # стоп >= 1.5 * ATR(15m) (обновлено v23)

    # --- ИСПОЛНЕНИЕ (FIX-1) ---
    "RESPECT_ORDER_TYPE": True,
    "MAX_MARKET_DRIFT_BPS": 12.0,  # дальше — Market-вход отменяется

    # === MAKER_ENTRY_PATCH_V35 ===
    # Все входы — PostOnly-лимит по лучшей цене своей стороны стакана
    # (bid для лонга, ask для шорта), поэтому вход всегда maker: 0.036%
    # вместо 0.100%. Market-сигналы Momentum ставятся по bid/ask и живут
    # MAKER_MARKET_TTL_SEC, затем снимаются как обычный таймаут.
    # False = поведение v34 (Momentum по рынку, лимитки могут пересечь стакан).
    "MAKER_ENTRY": True,
    "MAKER_MARKET_TTL_SEC": 180,

    "WALL_MULTIPLIER": 2.3,
    "OB_DEPTH": 90,
    "MAX_SPREAD_PCT": 0.09,
    # === ORDER_CANCEL_DISTANCE_PATCH_V11 ===
    "ORDER_CANCEL_DISTANCE": 0.015,   # 1.5% дистанция отмены (было 1%)
    # === R_TRAIL_PATCH_V23 (заменяет TRAIL_BY_PCT_PATCH_V22) ===
    # === EARLY_BE_R_TUNING_PATCH_V25 ===
    "EARLY_BE_R": 0.3,
    # === TRAIL_ACTIVATION_TUNING_PATCH_V33 ===
    # Было 1.2. За 7 сделок с телеметрией MFE максимум = 1.082R -> 0 активаций.
    # Порог стоял выше всего, что рынок давал: механизм был выключен.
    # 0.8 при TRAIL_DIST_R=0.35 даёт стартовый стоп 0.45R — заметно выше
    # уровня безубытка (+0.04..+0.14R). Ниже 0.5 идти нельзя: стоп сравнялся
    # бы с БУ, при 0.3 ушёл бы в минус.
    "TRAIL_ACTIVATION_R": 0.8,
    # === TRAIL_DIST_TIGHTEN_PATCH_V30 ===
    "TRAIL_DIST_R": 0.35,
    "ORDER_TIMEOUT_MINUTES": 15,      # таймаут лимитки, проверяется в smart_pending_manager

    # --- ИНСТИТУЦИОНАЛЬНЫЕ НАСТРОЙКИ ---
    "MAX_DAILY_ATR_EXHAUSTION": 0.75, # Лимит дневного хода
    "SL_NOISE_OFFSET_ATR": 0.15,      # Поправка на шум (0.15 ATR)
    "MIN_INSTITUTIONAL_RR": 3.0,      # Жесткий R:R не менее 1:3
    "DBSCAN_EPS_ATR": 0.2,            # Радиус склейки уровней
    # ------------------------------------------------

    "MIN_LIQUIDITY_DENSITY": 1.5,

    # --- MOMENTUM BREAKOUT (новая стратегия) ---
    "MOMENTUM_LOOKBACK": 20,
    "MOMENTUM_VOL_MULT": 1.5,
    "MOMENTUM_MIN_RR": 2.0,
    "MOMENTUM_BODY_RATIO": 0.6,
    "MOMENTUM_MAX_WICK_RATIO": 0.35,

    # --- ИНДИВИДУАЛЬНЫЕ ПОРОГИ ВХОДА ПО СТРАТЕГИЯМ ---
    "STRATEGY_MIN_SCORE": {
        "InstitutionalSMCPRO": 75,
        "TrendPRO": 65,
        "MomentumBreakoutPRO": 65,
        "DEFAULT": 75
    },
}

# ==========================================
# 🔕 PRODUCTION LOGGER
# ==========================================
class NotificationManager:
    DEBUG = False
    IMPORTANT = {'ENTRY', 'FILLED', 'TP', 'SL', 'CANCEL', 'ERROR', 'WARNING', 'START'}

    def __init__(self):
        self.last_messages = {}
        self.cooldown = 60

    def send(self, symbol, level, text):
        now = time.time()
        key = f'{symbol}:{level}'
        last = self.last_messages.get(key, 0)
        
        if level != 'START' and now - last < self.cooldown:
            return
            
        self.last_messages[key] = now
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        console_msg = f'[{timestamp}] [{level}] {symbol} | {text.replace(chr(10), " ")}'
        
        if self.DEBUG: print(console_msg)
            
        if level in self.IMPORTANT:
            if not self.DEBUG: print(console_msg)
            send_telegram(text)

notify = NotificationManager()

# ==========================================
# 🔐 SECRETS & ENV VARS
# ==========================================
# Ключи живут в predator_secrets.py рядом с этим файлом (4 строки вида
# API_KEY = "..."), чтобы код можно было хранить и пересылать без них.
try:
    from predator_secrets import API_KEY, API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    API_KEY = API_SECRET = TELEGRAM_BOT_TOKEN = TELEGRAM_CHAT_ID = "ВСТАВЬТЕ"

if API_KEY.startswith("ВСТАВЬТЕ"):
    print("ОШИБКА: не найден predator_secrets.py с ключами Bybit/Telegram.")
    sys.exit(1)

session = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)

# ==========================================
# 💾 STATE PERSISTENCE
# ==========================================
STATE_FILE = "bot_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f: return json.load(f)
        except Exception: return {}
    return {}

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=4)
    except Exception as e:
        log_msg("GLOBAL", f"⚠ Ошибка сохранения состояния: {e}", "ERROR")

state_db = load_state()

def api_retry(max_retries=3, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try: return func(*args, **kwargs)
                except Exception as e:
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except Exception: pass

def log_msg(symbol, message, level='INFO'):
    now = time.time()
    global TG_ANTI_SPAM
    if 'TG_ANTI_SPAM' not in globals(): globals()['TG_ANTI_SPAM'] = {}
    if len(TG_ANTI_SPAM) > 5000: TG_ANTI_SPAM.clear()
    
    ref_match = re.search(r'ID:\s*`?([a-zA-Z0-9-]+)`?', message)
    ref_id = ref_match.group(1) if ref_match else 'global'
    
    event_type = 'INFO'
    if 'Ордер принят' in message: event_type = 'ENTRY'
    elif 'исполнен' in message: event_type = 'FILLED'
    elif 'Take Profit' in message or 'TAKE PROFIT' in message or 'TRAILING STOP' in message: event_type = 'TP'
    elif 'Stop Loss' in message or 'STOP LOSS' in message: event_type = 'SL'
    elif 'отменён' in message or 'SMART TIME EXIT' in message or 'СТРУКТУРНЫЙ ВЫХОД' in message: event_type = 'CANCEL'
    elif 'Недостаточно' in message: event_type = 'MARGIN'
    elif 'Отказ биржи' in message or 'Error' in message or '⚠' in message: event_type = 'ERROR'
    
    key = f'{symbol}_{event_type}_{ref_id}'
    if event_type in ['MARGIN', 'ERROR']: cooldown = 900
    elif event_type in ['ENTRY', 'FILLED', 'CANCEL', 'TP', 'SL']: cooldown = 86400 * 365
    else: cooldown = 0
    
    if cooldown > 0 and now - TG_ANTI_SPAM.get(key, 0) < cooldown: return
    if cooldown > 0: TG_ANTI_SPAM[key] = now
    
    if event_type == 'MARGIN':
        if symbol not in state_db: state_db[symbol] = {}
        state_db[symbol]['cooldown_until'] = now + 900
        save_state(state_db)
        FSM.set_state('SCAN', None, 'MARGIN_ERROR')
    
    allowed = ['ENTRY', 'FILLED', 'TP', 'SL', 'CANCEL', 'MARGIN', 'ERROR', 'WARNING']
    if event_type in allowed or level == 'WARNING': _log_msg_raw(symbol, message, level)

def _log_msg_raw(symbol, message, level='INFO'):
    notify.send(symbol, level, message)

def diag(symbol, message):
    """Диагностика решений бота — прямо в консоль, минуя фильтры log_msg.

    log_msg классифицирует сообщение по ключевым словам и отдаёт в Telegram
    только ENTRY/FILLED/TP/SL/CANCEL/MARGIN/ERROR. Всё остальное молча
    отбрасывается, а совпадение со словом-триггером (напр. "отменён")
    вешает антиспам-кулдаун на год. Причины отказа от входа обязаны быть
    видны всегда и не должны попадать в Telegram — поэтому отдельный канал.
    """
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [DIAG] {symbol} | {message}")

# ==========================================
# 📊 DATA CLASSES
# ==========================================
@dataclass
class StrategySignal:
    strategy_name: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float        
    risk_reward: float
    prob_success: float
    structure_score: float
    liquidity_score: float
    reason: str
    state: str = "VALID"
    # --- НОВЫЕ ПОЛЯ ДЛЯ КОНТЕКСТА ---
    context_mode: str = "FREE"
    risk_modifier: float = 1.0
    d1_trend: str = "FLAT"
    h1_trend: str = "FLAT"
    m15_trend: str = "FLAT"
    order_type: str = "Limit" # === EXECUTION_FIX_PATCH_V17 ===

    def get_final_score(self) -> float:
        rr_normalized = min(100.0, max(0.0, (self.risk_reward / 4.0) * 100))
        return (self.confidence * 0.40) + (rr_normalized * 0.25) + (self.structure_score * 0.20) + (self.liquidity_score * 0.15)

class BaseStrategy:
    def analyze(self, symbol: str, dfs: Dict[str, pd.DataFrame], ob_data: dict, winrate: float) -> Optional[StrategySignal]:
        raise NotImplementedError

# ==========================================
# 🕒 MULTI-TIMEFRAME CACHE
# ==========================================
class MultiTimeframeCache:
    def __init__(self, session):
        self.session = session
        self.cache = {}

    @api_retry(max_retries=3, delay=1.0)
    def fetch(self, symbol, interval, cache_ttl):
        now = time.time()
        if symbol not in self.cache: self.cache[symbol] = {}
        if interval in self.cache[symbol]:
            cached_df, cached_time = self.cache[symbol][interval]
            if now - cached_time < cache_ttl: return cached_df

        limit = 200 if interval in ['240', '60', 'D'] else 100
        res = self.session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
        if not res or res.get('retCode') != 0: return None

        k_list = res['result']['list']
        k_list.reverse()
        df = pd.DataFrame(k_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)

        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        # ИСПРАВЛЕНИЕ 1: Оптика EMA200
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        
        df['chop'] = ta.chop(df['high'], df['low'], df['close'], length=14)

        self.cache[symbol][interval] = (df, now)
        return df

# ==========================================
# 🧱 СТРУКТУРНЫЙ МОДУЛЬ
# ==========================================
class StructureDetector:
    def __init__(self, symbol, swing_window=2, atr_period=14):
        self.symbol = symbol
        self.n = swing_window

    def analyze_structure(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        local_max_idx = argrelextrema(df['high'].values, np.greater, order=self.n)[0]
        local_min_idx = argrelextrema(df['low'].values, np.less, order=self.n)[0]

        df['Swing'] = None
        df.iloc[local_max_idx, df.columns.get_loc('Swing')] = 'Peak'
        df.iloc[local_min_idx, df.columns.get_loc('Swing')] = 'Trough'
        candidates = df.dropna(subset=['Swing'])

        last_peak, last_trough = None, None
        trend = "FLAT"
        last_event = "INIT"
        
        for idx, row in candidates.iterrows():
            cand_type = row['Swing']
            price = row['high'] if cand_type == 'Peak' else row['low']
            if cand_type == 'Peak':
                if last_peak is not None and price > last_peak:
                    last_event = "BOS_UP" if trend == "BULL" else "CHOCH_UP"
                    trend = "BULL"
                last_peak = price
            elif cand_type == 'Trough':
                if last_trough is not None and price < last_trough:
                    last_event = "BOS_DOWN" if trend == "BEAR" else "CHOCH_DOWN"
                    trend = "BEAR"
                last_trough = price
        return {"trend": trend, "event": last_event, "last_peak": last_peak, "last_trough": last_trough}

# ==========================================
# 🧠 TRADECONTEXT ENGINE 
# ==========================================
class TradeContextEngine:
    @staticmethod
    def analyze(dfs: Dict[str, pd.DataFrame], signal_dir: str, symbol: str):
        df_d1 = dfs.get("D")
        df_h1 = dfs.get("60")
        df_m15 = dfs.get("15")
        
        if df_d1 is None or df_h1 is None or df_m15 is None:
            return "NO TRADE", 0.0, "FLAT", "FLAT", "FLAT"
            
        detector = StructureDetector(symbol)
        d1_trend = detector.analyze_structure(df_d1).get('trend', 'FLAT')
        h1_trend = detector.analyze_structure(df_h1).get('trend', 'FLAT')
        m15_trend = detector.analyze_structure(df_m15).get('trend', 'FLAT')
        
        d1_dir = "Long" if d1_trend == "BULL" else "Short" if d1_trend == "BEAR" else "FLAT"
        h1_dir = "Long" if h1_trend == "BULL" else "Short" if h1_trend == "BEAR" else "FLAT"
        
        if d1_dir != "FLAT" and h1_dir != "FLAT" and d1_dir != h1_dir:
            return "NO TRADE", 0.0, d1_trend, h1_trend, m15_trend
            
        if d1_dir == h1_dir and d1_dir != "FLAT":
            if signal_dir == d1_dir:
                return "FREE", 1.0, d1_trend, h1_trend, m15_trend
            else:
                return "NERVOUS", 0.5, d1_trend, h1_trend, m15_trend
                
        return "FREE", 1.0, d1_trend, h1_trend, m15_trend

# ==========================================
# 🧠 INSTITUTIONAL S&R ENGINE
# ==========================================
class SREngine:
    def __init__(self, atr_period=14, dbscan_eps_atr=CONFIG["DBSCAN_EPS_ATR"]):
        self.atr_period = atr_period
        self.dbscan_eps_atr = dbscan_eps_atr

    def detect_trend_breaks(self, df: pd.DataFrame, left_bars=3, right_bars=3) -> list:
        levels = []
        if len(df) < left_bars + right_bars + 1: return levels
        
        for i in range(left_bars, len(df) - right_bars):
            is_high = True
            for j in range(1, left_bars + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j]: is_high = False
            for j in range(1, right_bars + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i + j]: is_high = False
            
            is_low = True
            for j in range(1, left_bars + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j]: is_low = False
            for j in range(1, right_bars + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i + j]: is_low = False

            if is_high: levels.append({'price': df['high'].iloc[i], 'type': 'RESISTANCE', 'index': i})
            if is_low: levels.append({'price': df['low'].iloc[i], 'type': 'SUPPORT', 'index': i})
        return levels

    def cluster_and_merge_levels(self, df: pd.DataFrame, raw_levels: list) -> list:
        if not raw_levels: return []
        atr = df['atr'].iloc[-1]
        prices = np.array([x['price'] for x in raw_levels]).reshape(-1, 1)

        eps_val = atr * self.dbscan_eps_atr
        if eps_val <= 0: eps_val = df['close'].iloc[-1] * 0.001

        db = DBSCAN(eps=eps_val, min_samples=1).fit(prices)
        labels = db.labels_

        merged_levels = []
        for cluster_id in set(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_elements = [raw_levels[i] for i in cluster_indices]
            
            mean_price = np.mean([x['price'] for x in cluster_elements])
            types = [x['type'] for x in cluster_elements]
            main_type = max(set(types), key=types.count)
            weight = len(cluster_elements)

            merged_levels.append({
                'price': mean_price,
                'type': main_type,
                'weight': weight,
                'min_price': np.min([x['price'] for x in cluster_elements]),
                'max_price': np.max([x['price'] for x in cluster_elements])
            })
        return merged_levels

# ==========================================
# 🕵️‍♂️ FALSE BREAKOUT ENGINE
# ==========================================
class FalseBreakoutEngine:
    def __init__(self, df: pd.DataFrame, level_price: float, level_type: str):
        self.df = df
        self.level = level_price
        self.type = level_type
        self.atr = df['atr'].iloc[-1]

    def check_1bar_false_breakout(self) -> dict:
        last_bar = self.df.iloc[-1]
        is_pattern = False
        metrics = {}

        if self.type == 'RESISTANCE':
            # === FALSE_BREAKOUT_TIGHTEN_PATCH_V29 ===
            if last_bar['high'] > self.level and last_bar['close'] < self.level:
                tail = last_bar['high'] - max(last_bar['open'], last_bar['close'])
                total_range = last_bar['high'] - last_bar['low']
                min_tail = CONFIG.get("SHADOW_MIN_TAIL_ATR_MULT", 0.3) * self.atr
                if tail >= min_tail and tail < 1.5 * self.atr: 
                    is_pattern = True
                    metrics = {'stop_loss': last_bar['high'] + (0.15 * self.atr), 'tail_size': tail}

        elif self.type == 'SUPPORT':
            if last_bar['low'] < self.level and last_bar['close'] > self.level:
                tail = min(last_bar['open'], last_bar['close']) - last_bar['low']
                min_tail = CONFIG.get("SHADOW_MIN_TAIL_ATR_MULT", 0.3) * self.atr
                if tail >= min_tail and tail < 1.5 * self.atr:
                    is_pattern = True
                    metrics = {'stop_loss': last_bar['low'] - (0.15 * self.atr), 'tail_size': tail}

        return {'detected': is_pattern, 'metrics': metrics}

    def check_2bar_false_breakout(self) -> dict:
        if len(self.df) < 2: return {'detected': False, 'metrics': {}}
        bar_1 = self.df.iloc[-2]
        bar_2 = self.df.iloc[-1]
        is_pattern = False
        metrics = {}

        if self.type == 'RESISTANCE':
            if bar_1['close'] > self.level and bar_2['open'] > self.level and bar_2['close'] < self.level:
                vol_confirmed = bar_2['volume'] > bar_1['volume']
                if vol_confirmed:
                    is_pattern = True
                    metrics = {
                        'stop_loss': max(bar_1['high'], bar_2['high']) + (0.15 * self.atr),
                        'volume_confirmation': vol_confirmed
                    }
        elif self.type == 'SUPPORT':
            if bar_1['close'] < self.level and bar_2['open'] < self.level and bar_2['close'] > self.level:
                vol_confirmed = bar_2['volume'] > bar_1['volume']
                if vol_confirmed:
                    is_pattern = True
                    metrics = {
                        'stop_loss': min(bar_1['low'], bar_2['low']) - (0.15 * self.atr),
                        'volume_confirmation': vol_confirmed
                    }
        return {'detected': is_pattern, 'metrics': metrics}

class AdvancedOrderBookAnalyzer:
    @staticmethod
    def analyze(bids, asks, wall_price, wall_vol, side):
        if not bids or not asks: return {"density": 0.0, "spoof_prob": 100, "real_liquidity_confirmed": False}
        surrounding_vol = 0.0
        wall_price_float = float(wall_price)
        book = bids if side == "Long" else asks
        for item in book:
            p, v = float(item[0]), float(item[1])
            if abs(p - wall_price_float) / wall_price_float <= 0.0015:
                if p != wall_price_float: surrounding_vol += v
        density = surrounding_vol / wall_vol if wall_vol > 0 else 0
        return {
            "density": density,
            "spoof_prob": 80 if density < 0.1 else max(0, 40 - (density * 20)),
            "real_liquidity_confirmed": density >= (CONFIG["MIN_LIQUIDITY_DENSITY"] / 10)
        }

class HighConfidenceDecisionEngine:
    def __init__(self, symbol, df_h4, df_h1, df_m15, df_1d, struct_h4, struct_h1, struct_m15,
                 spread, risk_pct, rr_ratio, sniper_dir, winrate, ob_analysis):
        self.symbol = symbol
        self.df_h4, self.struct_h4 = df_h4, struct_h4
        self.df_h1, self.struct_h1 = df_h1, struct_h1
        self.df_m15, self.struct_m15 = df_m15, struct_m15
        self.df_1d = df_1d 
        self.spread = spread
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.sniper_dir = sniper_dir
        self.winrate = winrate
        self.ob_analysis = ob_analysis
        self.reasons = []

    def _check_atr_exhaustion(self):
        if self.df_1d is not None and not self.df_1d.empty and self.df_m15 is not None:
            daily_candle = self.df_1d.iloc[-1]
            daily_open = daily_candle['open']
            daily_atr = daily_candle['atr']
            current_price = self.df_m15.iloc[-1]['close']
            travel = abs(current_price - daily_open)
            limit = daily_atr * CONFIG["MAX_DAILY_ATR_EXHAUSTION"]
            if travel > limit:
                return False
        return True

    def evaluate(self):
        score = 60 
        if not self._check_atr_exhaustion(): return "SKIP", score, self.reasons
        if self.spread > CONFIG["MAX_SPREAD_PCT"]: return "SKIP", score, ["Spread High"]

        if self.df_h1 is not None and not self.df_h1.empty:
            close_price = self.df_h1['close'].iloc[-1]
            ema_200 = self.df_h1['ema_200'].iloc[-1]
            if self.sniper_dir == "Long" and close_price < ema_200:
                return "SKIP", score, self.reasons
            if self.sniper_dir == "Short" and close_price > ema_200:
                return "SKIP", score, self.reasons

        if self.ob_analysis['spoof_prob'] > 50: score -= 15
        if score >= 60: return "ENTRY_ALLOWED", score, ["Passed"]
        return "SKIP", score, self.reasons

# ==========================================
# ⚙️ ПОЛНОЦЕННЫЙ ADAPTIVE POSITION MANAGER
# ==========================================
class AdaptivePositionManager:
    # === TRUE_RISK_DISTANCE_FIX_PATCH_V24 ===
    def __init__(self, df_m15, df_m5, entry_price, current_price, current_sl, initial_sl, side, symbol, mode="FREE", true_entry_time=None, is_early_be_set=False, original_risk_distance=None):
        self.df = df_m15 
        self.df_m5 = df_m5
        self.entry_price = float(entry_price)
        self.current_price = float(current_price)
        self.current_sl = float(current_sl) if current_sl else 0.0
        self.initial_sl = float(initial_sl) if initial_sl else self.current_sl
        self.side = side
        self.is_early_be_set = is_early_be_set
        self.symbol = symbol
        self.mode = mode
        self.true_entry_time = true_entry_time
        self.original_risk_distance = original_risk_distance

    def analyze(self):
        if self.df is None or len(self.df) < 10 or 'atr' not in self.df.columns: 
            return 'NONE', self.current_sl, "", 0.0, 0.0, ""
        
        atr = self.df['atr'].iloc[-1]
        
        # 1. R Engine (Двигатель рисков)
        if self.original_risk_distance and self.original_risk_distance > 0:
            initial_risk = self.original_risk_distance
        else:
            initial_risk = abs(self.entry_price - self.initial_sl)
        if initial_risk == 0: initial_risk = atr # Fallback
        
        if self.side == "Buy":
            pnl_dist = self.current_price - self.entry_price
        else:
            pnl_dist = self.entry_price - self.current_price
            
        current_r = pnl_dist / initial_risk

        # --- БЛОК: СТРУКТУРНЫЙ ВЫХОД (УМНЫЙ ДЕТЕКТОР УГАСАНИЯ) ---
        if self.df_m5 is not None and not self.df_m5.empty:
            c1 = self.df_m5.iloc[-1]
            
            trend_fade_ema = False
            if self.side == 'Buy' and c1['close'] < c1['ema_20']: trend_fade_ema = True
            if self.side == 'Sell' and c1['close'] > c1['ema_20']: trend_fade_ema = True
            
            m5_struct = StructureDetector(self.symbol).analyze_structure(self.df_m5)
            m5_event = m5_struct.get('event', '')
            bos_against = False
            if self.side == 'Buy' and m5_event in ['BOS_DOWN', 'CHOCH_DOWN']: bos_against = True
            if self.side == 'Sell' and m5_event in ['BOS_UP', 'CHOCH_UP']: bos_against = True
            
            vol_dropped = c1['volume'] < c1['vol_sma']
            
            # === POSITION_MGMT_TUNING_PATCH_V8 ===
            structural_signal_count = sum([trend_fade_ema, bos_against, vol_dropped])
            seconds_in_trade = (time.time() - self.true_entry_time) if self.true_entry_time else 999999
            min_time_ok = seconds_in_trade >= CONFIG.get("MIN_TIME_IN_TRADE_SECONDS", 600)
            pct_against = abs(self.current_price - self.entry_price) / self.entry_price if self.entry_price else 0
            min_pct_ok = pct_against >= CONFIG.get("MIN_STRUCTURAL_EXIT_PCT", 0.006)
            if structural_signal_count >= 2 and min_time_ok and current_r < 0 and min_pct_ok:
                reason_msg = f"🛑 СТРУКТУРНЫЙ ВЫХОД\nМонета: {self.symbol}\nРежим: {self.mode}\nПричина:\n- Пробита EMA20 (M5)\n- Слом BOS против позиции\n- Падение объемов\nСделка экстренно закрыта!"
                return 'TIME_EXIT', 0.0, reason_msg, current_r, atr, "STRUCTURAL_EXIT"

        # 2. Истинный безубыток (True BE) с учетом комиссий
        fee_maker = self.entry_price * 0.0002
        fee_taker = self.entry_price * 0.00055
        slippage_buffer = atr * 0.05
        tick_buffer = self.entry_price * 0.0005
        
        total_cost_offset = fee_maker + fee_taker + slippage_buffer + tick_buffer
        
        true_be = self.entry_price + total_cost_offset if self.side == 'Buy' else self.entry_price - total_cost_offset

        # === EXECUTION_FIX_PATCH_V17 ===
        if current_r >= CONFIG.get("EARLY_BE_R", 1.0) and not self.is_early_be_set:
            return 'SET_EARLY_BE', true_be, "", current_r, atr, "1R_EARLY_BE"

        # 3. State Machine
        action = 'NONE'
        new_sl = self.current_sl
        reason_str = ""

        # === R_TRAIL_PATCH_V23 (заменяет TRAIL_BY_PCT_PATCH_V22) ===
        trail_activation_r = CONFIG.get("TRAIL_ACTIVATION_R", 1.2)
        trail_dist_r = CONFIG.get("TRAIL_DIST_R", 0.6)

        if current_r >= trail_activation_r:
            action = 'TRAIL'
            trail_dist_abs = initial_risk * trail_dist_r
            new_sl = self.current_price - trail_dist_abs if self.side == "Buy" else self.current_price + trail_dist_abs
            reason_str = "R_TRAIL"

        if action != 'NONE':
            return action, new_sl, "", current_r, atr, reason_str
        
        return 'NONE', self.current_sl, "", current_r, atr, ""

# ==========================================
# 📈 СТРАТЕГИИ
# ==========================================
class LiquidityTrendPRO(BaseStrategy):
    def analyze(self, symbol: str, dfs: Dict[str, pd.DataFrame], ob_data: dict, winrate: float) -> Optional[StrategySignal]:
        if not ob_data or ob_data.get('sniper_score', 0) == 0: return None
        df_h4, df_h1, df_m15, df_1d = dfs.get("240"), dfs.get("60"), dfs.get("15"), dfs.get("D")
        if df_h4 is None or df_h1 is None or df_m15 is None or df_1d is None: return None

        sniper_dir = ob_data['dir']
        entry_price = ob_data['price']
        spread_pct = (ob_data['ask'] - ob_data['bid']) / ob_data['bid'] * 100

        struct_det = StructureDetector(symbol)
        struct_m15 = struct_det.analyze_structure(df_m15)
        ob_analysis = AdvancedOrderBookAnalyzer.analyze(ob_data['bids'], ob_data['asks'], entry_price, ob_data['vol'], sniper_dir)
        atr_val = df_m15['atr'].iloc[-1]

        # === TRENDPRO_CONFIRMATION_PATCH ===
        # Раньше TrendPRO входил по одному снимку стакана (крупнейшая стенка) без единой
        # проверки, что цена вообще отреагировала на этот уровень. Остальные две живые
        # стратегии (SMC, Momentum) так не делают — обе требуют реального подтверждения
        # ценой/объёмом. Ниже - те же 4 проверки, что уже работают у них, приведённые к
        # единому стандарту:

        # 1. Реальная ликвидность вокруг стенки (метрика уже считалась в ob_analysis,
        #    но нигде не проверялась — стенка могла быть спуфом без всякого фильтра)
        if not ob_analysis.get('real_liquidity_confirmed', False):
            return None

        # 2. Объёмный всплеск на последней M15-свече — ровно тот же порог, что у SMC (1.15x)
        vol_surge = df_m15['volume'].iloc[-1] > (df_m15['vol_sma'].iloc[-1] * 1.15)
        if not vol_surge:
            return None

        # 3. Фильтр choppy-рынка — тот же порог 61.8, что у SMC и Momentum
        current_chop = df_m15['chop'].iloc[-1]
        if not pd.isna(current_chop) and current_chop > 61.8:
            return None

        # 4. Подтверждение ЦЕНОЙ: требуем, чтобы цена уже протестировала уровень стенки
        #    и отреагировала (ложный пробой) — тот же движок FalseBreakoutEngine, что
        #    использует InstitutionalSMCPRO, применённый к цене стенки как к уровню.
        level_type = 'SUPPORT' if sniper_dir == 'Long' else 'RESISTANCE'
        fb_engine = FalseBreakoutEngine(df_m15, entry_price, level_type)
        fb_1b = fb_engine.check_1bar_false_breakout()
        fb_2b = fb_engine.check_2bar_false_breakout()
        if not (fb_1b['detected'] or fb_2b['detected']):
            return None

        noise_offset = atr_val * CONFIG["SL_NOISE_OFFSET_ATR"]
        
        if sniper_dir == "Long":
            local_low = struct_m15.get('last_trough') or (entry_price - atr_val)
            sl_price = min(entry_price - noise_offset, local_low - noise_offset)
            tp_price = entry_price + (abs(entry_price - sl_price) * 3.0) 
        else:
            local_high = struct_m15.get('last_peak') or (entry_price + atr_val)
            sl_price = max(entry_price + noise_offset, local_high + noise_offset)
            tp_price = entry_price - (abs(sl_price - entry_price) * 3.0)

        risk_dist = abs(entry_price - sl_price) / entry_price * 100
        rr_ratio = abs(tp_price - entry_price) / abs(entry_price - sl_price) if abs(entry_price - sl_price) > 0 else 0

        engine = HighConfidenceDecisionEngine(
            symbol, df_h4, df_h1, df_m15, df_1d, {}, {}, struct_m15,
            spread_pct, risk_dist, rr_ratio, sniper_dir, winrate, ob_analysis
        )
        decision, final_score, reasons = engine.evaluate()

        if decision == "ENTRY_ALLOWED":
            return StrategySignal("TrendPRO", symbol, sniper_dir, entry_price, sl_price, tp_price,
                final_score, rr_ratio, winrate or 50.0, final_score, 100, "Trend Confirmed")
        return None

class InstitutionalSMCPRO(BaseStrategy):
    def __init__(self):
        self.sr_engine = SREngine()

    def analyze(self, symbol: str, dfs: Dict[str, pd.DataFrame], ob_data: dict, winrate: float) -> Optional[StrategySignal]:
        df_d1 = dfs.get("D")
        df_h1 = dfs.get("60")
        df_m15 = dfs.get("15")
        
        if df_d1 is None or df_h1 is None or df_m15 is None or len(df_h1) < 50: return None

        raw_levels_d1 = self.sr_engine.detect_trend_breaks(df_d1, left_bars=3, right_bars=3)
        raw_levels_h1 = self.sr_engine.detect_trend_breaks(df_h1, left_bars=5, right_bars=5)
        merged_zones = self.sr_engine.cluster_and_merge_levels(df_h1, raw_levels_d1 + raw_levels_h1)

        current_price = df_m15['close'].iloc[-1]
        atr_m15 = df_m15['atr'].iloc[-1]

        closest_zone = None
        min_dist = float('inf')
        for zone in merged_zones:
            dist = abs(current_price - zone['price'])
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone

        if not closest_zone or min_dist > (atr_m15 * 3): 
            return None 

        lp_engine = FalseBreakoutEngine(df_m15, closest_zone['price'], closest_zone['type'])
        lp_1b = lp_engine.check_1bar_false_breakout()
        lp_2b = lp_engine.check_2bar_false_breakout()

        signal_dir = None
        sl_price = 0.0

        if lp_1b['detected']:
            signal_dir = "Long" if closest_zone['type'] == 'SUPPORT' else "Short"
            sl_price = lp_1b['metrics']['stop_loss']
        elif lp_2b['detected']:
            signal_dir = "Long" if closest_zone['type'] == 'SUPPORT' else "Short"
            sl_price = lp_2b['metrics']['stop_loss']

        try:
            ema_50 = df_m15['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            current_close = df_m15['close'].iloc[-1]
            if signal_dir == 'Long' and current_close < ema_50:
                signal_dir = None
            elif signal_dir == 'Short' and current_close > ema_50:
                signal_dir = None
        except Exception: pass

        vol_surge = df_m15['volume'].iloc[-1] > (df_m15['vol_sma'].iloc[-1] * 1.15)
        current_chop = df_m15['chop'].iloc[-1]
        is_choppy = current_chop > 61.8 if not pd.isna(current_chop) else False

        if is_choppy or not vol_surge:
            return None

        df_m5 = dfs.get("5")
        if df_m5 is not None and len(df_m5) >= 2 and signal_dir:
            c1, c2 = df_m5.iloc[-2], df_m5.iloc[-1]
            if signal_dir == 'Long' and c1['close'] < c1['open'] and c2['close'] < c2['open']:
                return None 
            if signal_dir == 'Short' and c1['close'] > c1['open'] and c2['close'] > c2['open']:
                return None

        if signal_dir:
            noise_offset = atr_m15 * CONFIG["SL_NOISE_OFFSET_ATR"]
            try:
                lookback = 15
                if signal_dir == 'Long':
                    sl_price = df_m15['low'].tail(lookback).min() - noise_offset
                elif signal_dir == 'Short':
                    sl_price = df_m15['high'].tail(lookback).max() + noise_offset
            except Exception: pass
            
            tp_dist = abs(current_price - sl_price) * CONFIG["MIN_INSTITUTIONAL_RR"]
            tp_price = current_price + tp_dist if signal_dir == "Long" else current_price - tp_dist
            rr = abs(tp_price - current_price) / abs(current_price - sl_price) if current_price != sl_price else 0

            if rr >= CONFIG["MIN_INSTITUTIONAL_RR"]:
                try:
                    if closest_zone['type'] == 'RESISTANCE' and len(df_m15) >= 3 and (df_m15['close'].iloc[-1] > df_m15['open'].iloc[-1]) and (df_m15['close'].iloc[-2] > df_m15['open'].iloc[-2]) and (df_m15['close'].iloc[-3] > df_m15['open'].iloc[-3]):
                        return None
                    if closest_zone['type'] == 'SUPPORT' and len(df_m15) >= 3 and (df_m15['close'].iloc[-1] < df_m15['open'].iloc[-1]) and (df_m15['close'].iloc[-2] < df_m15['open'].iloc[-2]) and (df_m15['close'].iloc[-3] < df_m15['open'].iloc[-3]):
                        return None
                except Exception: pass
                
                return StrategySignal(
                    strategy_name="InstitutionalSMCPRO", symbol=symbol, direction=signal_dir,
                    entry_price=current_price, stop_loss=sl_price, take_profit=tp_price,
                    confidence=95.0, risk_reward=rr, prob_success=winrate or 50.0,
                    structure_score=100, liquidity_score=100,
                    reason=f"False Breakout detected at {closest_zone['type']} zone"
                )
        return None

# === MOMENTUM_BREAKOUT_PATCH_V1 ===
class MomentumBreakoutPRO(BaseStrategy):
    """
    В отличие от InstitutionalSMCPRO (торгует ЛОЖНЫЕ пробои, разворот/fade),
    эта стратегия торгует ПОДТВЕРЖДЁННЫЕ пробои С ПРОДОЛЖЕНИЕМ (momentum),
    в сторону движения, а не против него.
    """
    def analyze(self, symbol: str, dfs: Dict[str, pd.DataFrame], ob_data: dict, winrate: float) -> Optional[StrategySignal]:
        df_h1 = dfs.get("60")
        df_m15 = dfs.get("15")
        lookback = CONFIG["MOMENTUM_LOOKBACK"]
        if df_h1 is None or df_m15 is None or len(df_m15) < lookback + 2:
            return None

        prior = df_m15.iloc[-(lookback + 1):-1]
        range_high = prior['high'].max()
        range_low = prior['low'].min()

        last = df_m15.iloc[-1]
        atr_m15 = df_m15['atr'].iloc[-1]
        if pd.isna(atr_m15) or atr_m15 <= 0:
            return None

        candle_range = last['high'] - last['low']
        if candle_range <= 0:
            return None
        body_ratio = abs(last['close'] - last['open']) / candle_range

        direction = None
        wick_ratio = 1.0
        if last['close'] > range_high and last['close'] > last['open']:
            direction = "Long"
            wick_ratio = (last['high'] - last['close']) / candle_range
        elif last['close'] < range_low and last['close'] < last['open']:
            direction = "Short"
            wick_ratio = (last['close'] - last['low']) / candle_range
        else:
            return None

        if body_ratio < CONFIG["MOMENTUM_BODY_RATIO"] or wick_ratio > CONFIG["MOMENTUM_MAX_WICK_RATIO"]:
            return None

        if not (last['volume'] > (df_m15['vol_sma'].iloc[-1] * CONFIG["MOMENTUM_VOL_MULT"])):
            return None

        chop_val = df_m15['chop'].iloc[-1]
        if not pd.isna(chop_val) and chop_val > 61.8:
            return None

        try:
            ema200_h1 = df_h1['ema_200'].iloc[-1]
            close_h1 = df_h1['close'].iloc[-1]
            if direction == "Long" and close_h1 < ema200_h1:
                return None
            if direction == "Short" and close_h1 > ema200_h1:
                return None
        except Exception:
            pass

        entry_price = last['close']
        noise_offset = atr_m15 * CONFIG["SL_NOISE_OFFSET_ATR"]

        if direction == "Long":
            sl_price = min(last['low'], range_high) - noise_offset
        else:
            sl_price = max(last['high'], range_low) + noise_offset

        risk = abs(entry_price - sl_price)
        if risk <= 0:
            return None

        rr = CONFIG["MOMENTUM_MIN_RR"]
        tp_price = entry_price + risk * rr if direction == "Long" else entry_price - risk * rr
        confidence = 70.0 + min(20.0, (body_ratio - CONFIG["MOMENTUM_BODY_RATIO"]) * 50)

        sig = StrategySignal(
            strategy_name="MomentumBreakoutPRO", symbol=symbol, direction=direction,
            entry_price=entry_price, stop_loss=sl_price, take_profit=tp_price,
            confidence=confidence, risk_reward=rr, prob_success=winrate or 50.0,
            structure_score=80, liquidity_score=70,
            reason=f"Momentum breakout of {lookback}-bar range with volume surge"
        )
        sig.order_type = "Market" # === EXECUTION_FIX_PATCH_V17 ===
        return sig

def _stop_distance_ok(signal, dfs_closed, symbol) -> bool:
    """FIX-2. Стоп обязан быть шире и рыночного шума, и издержек круга."""
    entry = float(signal.entry_price)
    if entry <= 0:
        return False
    risk_bps = abs(entry - float(signal.stop_loss)) / entry * 1e4

    floor_bps = max(CONFIG["MIN_STOP_DISTANCE_BPS"],
                    CONFIG["COST_ROUND_TRIP_BPS"] * CONFIG["MIN_EDGE_COST_MULT"])
    try:
        df15 = dfs_closed.get("15")
        if df15 is not None and not df15.empty:
            atr = float(df15["atr"].iloc[-1])
            if atr > 0:
                floor_bps = max(floor_bps, atr / entry * 1e4 * CONFIG["MIN_STOP_ATR_MULT"])
    except Exception:
        pass

    if risk_bps < floor_bps:
        diag(symbol, f"COST-GUARD: {signal.strategy_name} отклонён — "
                     f"стоп {risk_bps:.1f}bps < порог {floor_bps:.1f}bps")
        return False
    return True


class StrategyManager:
    def __init__(self):
        self.strategies: List[BaseStrategy] = [
            LiquidityTrendPRO(),
            InstitutionalSMCPRO(),
            MomentumBreakoutPRO()
        ]

    def evaluate_all(self, symbol: str, dfs: Dict[str, pd.DataFrame], ob_data: dict, winrate: float) -> Optional[StrategySignal]:
        # === EXECUTION_FIX_PATCH_V17 === Anti-Repaint
        dfs_closed = {k: (v.iloc[:-1].copy() if v is not None and len(v)>1 else v) for k, v in dfs.items()}
        valid_signals = []
        for strategy in self.strategies:
            try:
                signal = strategy.analyze(symbol, dfs_closed, ob_data, winrate)
                if signal and signal.state == "VALID":
                    # --- FIX-2: пол стоп-дистанции. Сигнал со стопом внутри
                    # шумовой полосы или уже дороже комиссии — не сигнал.
                    if not _stop_distance_ok(signal, dfs_closed, symbol):
                        continue
                    # --- FIX-4: контекст считаем ТОЛЬКО по закрытым свечам.
                    mode, risk_mod, d1_t, h1_t, m15_t = TradeContextEngine.analyze(dfs_closed, signal.direction, symbol)
                    if mode == "NO TRADE":
                        continue 
                    # === SKIP_NERVOUS_PATCH_V14 ===
                    if mode == "NERVOUS":
                        continue
                    signal.context_mode = mode
                    signal.risk_modifier = risk_mod
                    signal.d1_trend = d1_t
                    signal.h1_trend = h1_t
                    signal.m15_trend = m15_t
                    valid_signals.append(signal)
            except Exception as e:
                log_msg(symbol, f"⚠ Ошибка в стратегии: {e}", "ERROR")

        if not valid_signals: return None

        def _passes_own_threshold(s):
            min_score = CONFIG["STRATEGY_MIN_SCORE"].get(
                s.strategy_name, CONFIG["STRATEGY_MIN_SCORE"].get("DEFAULT", 75)
            )
            return s.get_final_score() >= min_score

        qualifying = [s for s in valid_signals if _passes_own_threshold(s)]
        if not qualifying: return None
        return max(qualifying, key=lambda s: s.get_final_score())

# ==========================================
# 📊 PERFORMANCE TRACKER
# ==========================================
class PerformanceTracker:
    def __init__(self, db_path="trades.db"):
        self.db_path = db_path
        self.active_trades = {}
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT, side TEXT, strategy TEXT,
                        entry_price REAL, size REAL, leverage REAL,
                        atr REAL, adx REAL, ema_dist REAL, structure TEXT,
                        market_state TEXT, ob_vol REAL, spread REAL, imbalance REAL,
                        sl REAL, tp REAL, risk_pct REAL, expected_rr REAL,
                        open_time DATETIME, close_time DATETIME,
                        close_price REAL, close_reason TEXT,
                        pnl_usdt REAL, pnl_pct REAL, r_multiple REAL,
                        max_mfe REAL, max_mae REAL, events_log TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            log_msg("GLOBAL", f"⚠ Ошибка инициализации БД: {e}", "ERROR")

    def on_position_opened(self, symbol, entry_price, size, side, signal: Optional[StrategySignal]):
        strat_name = signal.strategy_name if signal else 'Unknown'
        expected_rr = signal.risk_reward if signal else 0.0
        self.active_trades[symbol] = {
            'symbol': symbol, 'side': side, 'strategy': strat_name,
            'entry_price': entry_price, 'size': size, 'leverage': CONFIG["LEVERAGE"],
            'sl': signal.stop_loss if signal else 0.0, 'tp': signal.take_profit if signal else 0.0,
            'signal_entry_price': signal.entry_price if signal else entry_price,
            'risk_pct': expected_rr, 'expected_rr': expected_rr,
            'open_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'max_mfe': 0.0, 'max_mae': 0.0, 'pending_close_reason': None,
            # === ENTRY_CONTEXT_PATCH_V34 === читаем из state_db (переживает рестарт)
            'entry_ctx': dict(state_db.get(symbol, {}).get('entry_context') or {})
        }

    # === MFE_MAE_WIRING_PATCH_V31 ===
    def update_excursion(self, symbol, current_r):
        """Трекинг максимальной экскурсии в R. Вызывается из manage_position.
        Ничего не решает — только фиксирует, куда ходила цена."""
        if symbol not in self.active_trades:
            return
        try:
            r = float(current_r)
        except (TypeError, ValueError):
            return
        if r != r or r in (float('inf'), float('-inf')):  # NaN / inf guard
            return
        trade = self.active_trades[symbol]
        changed = None
        if r > trade.get('max_mfe', 0.0):
            trade['max_mfe'] = r
            changed = 'MFE'
        if r < trade.get('max_mae', 0.0):
            trade['max_mae'] = r
            changed = 'MAE'
        if changed:
            diag(symbol, f"[EXCURSION] {changed} -> {r:.2f}R "
                         f"(MFE={trade['max_mfe']:.2f}R MAE={trade['max_mae']:.2f}R)")

    def set_close_reason(self, symbol, reason):
        if symbol in self.active_trades:
            self.active_trades[symbol]['pending_close_reason'] = reason

    @api_retry(max_retries=3, delay=1.0)
    def check_closed_positions(self, current_active_symbols, bybit_session):
        closed_syms = [s for s in list(self.active_trades.keys()) if s not in current_active_symbols]
        for sym in closed_syms:
            trade = self.active_trades[sym]
            reason = trade.get('pending_close_reason') or 'Exchange TP/SL'
            pnl_usdt, close_price, pnl_pct = 0.0, 0.0, 0.0
            # === STALE_CLOSED_PNL_FIX_PATCH_V27 ===
            try:
                open_time_ms = None
                try:
                    open_time_dt = datetime.datetime.strptime(trade['open_time'], '%Y-%m-%d %H:%M:%S')
                    open_time_ms = open_time_dt.timestamp() * 1000
                except Exception:
                    pass

                data = None
                for attempt in range(5):
                    res = bybit_session.get_closed_pnl(category="linear", symbol=sym, limit=5)
                    if res and res.get('retCode') == 0:
                        records = res.get('result', {}).get('list', [])
                        for rec in records:
                            try:
                                rec_time = float(rec.get('updatedTime', 0))
                            except Exception:
                                rec_time = 0
                            if open_time_ms is None or rec_time >= open_time_ms - 5000:
                                data = rec
                                break
                    if data:
                        break
                    time.sleep(1.5)

                if data:
                    pnl_usdt = float(data.get('closedPnl', 0))
                    close_price = float(data.get('avgExitPrice', trade['entry_price']))
                    if trade['size'] > 0 and trade['entry_price'] > 0:
                        position_value = trade['size'] * trade['entry_price']
                        pnl_pct = (pnl_usdt / position_value) if position_value > 0 else 0
                else:
                    log_msg(sym, "⚠ Не найдена свежая запись closed_pnl за 5 попыток — "
                                 "PnL этой сделки НЕ записан достоверно (fallback: entry_price, PnL=0)",
                            "ERROR")
                    close_price = trade['entry_price']
                    pnl_usdt = 0.0
                    pnl_pct = 0.0
            except Exception as e:
                log_msg(sym, f"⚠ Ошибка получения закрытых PnL: {e}", "ERROR")
            self._finalize_trade(sym, trade, close_price, reason, pnl_usdt, pnl_pct)

    def _finalize_trade(self, symbol, trade, close_price, reason, pnl_usdt, pnl_pct):
        close_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate final R
        # === TRUE_RISK_DISTANCE_FIX_PATCH_V24 ===
        _signal_entry = trade.get('signal_entry_price', trade['entry_price'])
        initial_risk = abs(_signal_entry - trade['sl'])
        r_multiple = 0.0
        if initial_risk > 0:
            if trade['side'] == 'Buy':
                r_multiple = (close_price - trade['entry_price']) / initial_risk
            else:
                r_multiple = (trade['entry_price'] - close_price) / initial_risk
        
        if trade and not trade.get('tp_sl_notified'):
            # ИСПРАВЛЕНИЕ 4: Честная причина закрытия по PnL и логике
            status_emoji = "❓"
            status_text = "UNKNOWN"
            
            if reason in ['TIME_EXIT', 'STRUCTURAL_EXIT']:
                status_emoji = "🟡"
                status_text = "MANUAL / STRUCTURAL CLOSE"
            elif reason == 'MARGIN_ERROR':
                status_emoji = "⚠️"
                status_text = "LIQUIDATION / MARGIN"
            else:
                # Определяем причину по реальным результатам закрытой сделки
                tp_dist = abs(close_price - trade['tp']) if trade['tp'] else float('inf')
                entry_price = trade['entry_price']
                
                if pnl_usdt > 0:
                    if tp_dist / entry_price < 0.005: 
                        status_emoji = "🟢"
                        status_text = "TAKE PROFIT (TARGET)"
                    else:
                        status_emoji = "🔵"
                        status_text = "TRAILING STOP (PROFIT)"
                else:
                    status_emoji = "🔴"
                    status_text = "STOP LOSS"

            # === NOTIFY_FIX_AND_PCT_BE_PATCH_V10 ===
            unique_ref = int(time.time() * 1000)
            msg = f'{status_emoji} {status_text}\nМонета: {symbol}\nРезультат: {r_multiple:.2f}R\nПрибыль: {pnl_pct*100:.2f}%\nPnL: {pnl_usdt:.2f} USDT\nID: {unique_ref}'
            if "TAKE PROFIT" in status_text or "TRAILING STOP" in status_text:
                log_msg(symbol, msg, 'TP')
            else:
                log_msg(symbol, msg, 'SL')
            trade['tp_sl_notified'] = True

            if status_text in ("STOP LOSS", "MANUAL / STRUCTURAL CLOSE", "LIQUIDATION / MARGIN"):
                cooldown_seconds = 900
                st = state_db.setdefault(symbol, {})
                st['cooldown_until'] = time.time() + cooldown_seconds
                save_state(state_db)
            
        # === ENTRY_CONTEXT_PATCH_V34 ===
        _raw_ctx = trade.get('entry_ctx') or {}
        _ctx = {k: _raw_ctx.get(k) for k in CTX_KEYS}
        if not any(v is not None for v in _ctx.values()):
            diag(symbol, "CTX: контекст входа пуст (вероятно рестарт до входа)")
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (
                        symbol, side, strategy, entry_price, size, leverage,
                        atr, adx, ema_dist, structure, market_state, ob_vol, spread, imbalance,
                        sl, tp, risk_pct, expected_rr, open_time, close_time, close_price, close_reason,
                        pnl_usdt, pnl_pct, r_multiple, max_mfe, max_mae, events_log
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', (
                    trade['symbol'], trade['side'], trade['strategy'], trade['entry_price'], trade['size'], trade['leverage'],
                    _ctx['atr'], _ctx['adx'], _ctx['ema_dist'], _ctx['structure'],
                    _ctx['market_state'], _ctx['ob_vol'], _ctx['spread'], _ctx['imbalance'],
                    trade['sl'], trade['tp'], trade['risk_pct'], trade['expected_rr'],
                    trade['open_time'], close_time, close_price, reason,
                    pnl_usdt, pnl_pct, r_multiple, trade['max_mfe'], trade['max_mae'], None
                ))
                conn.commit()
        except Exception as e:
            log_msg(symbol, f"⚠ Ошибка записи в БД: {e}", "ERROR")
        if symbol in self.active_trades: del self.active_trades[symbol]

    def get_coin_stats(self, symbol):
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                df = pd.read_sql_query(f"SELECT pnl_usdt FROM trades WHERE symbol='{symbol}'", conn)
                if len(df) < 5: return None
                wins = df[df['pnl_usdt'] > 0]
                return (len(wins) / len(df)) * 100
        except Exception: return None

# === ENTRY_CONTEXT_PATCH_V34 ===
def snapshot_entry_context(symbol, signal, dfs, ob_data):
    """Снимок обстановки на момент сигнала. ТОЛЬКО телеметрия.

    Считается по ЗАКРЫТЫМ барам (как и сами сигналы, см. evaluate_all).
    Любая ошибка внутри -> словарь из None. Вход не должен падать из-за
    диагностики. Возвращает 8 ключей == 8 пустых колонок таблицы trades.

    Единицы:
      atr          ATR(14) M15, абсолютная цена (делить на entry_price для bps)
      adx          ADX(14) M15, пункты
      ema_dist     (entry - EMA200_M15)/EMA200_M15 * 1e4, bps, со знаком
      structure    событие структуры M15: BOS_UP/CHOCH_DOWN/INIT/...
      market_state CTX/D1/H1/M15/CHOP одной строкой, разделитель '|'
      ob_vol       объём крупнейшей стенки стакана (ob_data['vol'])
      spread       (ask-bid)/bid * 1e4, bps
      imbalance    (sum_bid - sum_ask)/(sum_bid + sum_ask), -1..+1
    """
    ctx = {"atr": None, "adx": None, "ema_dist": None, "structure": None,
           "market_state": None, "ob_vol": None, "spread": None, "imbalance": None}
    try:
        df = dfs.get("15") if dfs else None
        if df is not None and len(df) > 2:
            d = df.iloc[:-1]  # anti-repaint: последний бар не закрыт
            last = d.iloc[-1]
            try:
                v = float(last.get("atr"))
                if v == v:
                    ctx["atr"] = v
            except Exception:
                pass
            try:
                ema = float(last.get("ema_200"))
                ep = float(signal.entry_price)
                if ema > 0 and ep > 0:
                    ctx["ema_dist"] = (ep - ema) / ema * 1e4
            except Exception:
                pass
            try:
                adx_df = ta.adx(d["high"], d["low"], d["close"], length=14)
                if adx_df is not None and len(adx_df) > 0:
                    # ADX_14, а не ADXR_14_2: pandas_ta отдаёт обе, префикс
                    # "ADX" матчит и ту и другую. Опираться на порядок колонок
                    # нельзя - подчёркивание отсекает ADXR.
                    col = [c for c in adx_df.columns if c.upper().startswith("ADX_")]
                    if col:
                        v = float(adx_df[col[0]].iloc[-1])
                        if v == v:
                            ctx["adx"] = v
            except Exception:
                pass
            try:
                ctx["structure"] = StructureDetector(symbol).analyze_structure(d).get("event")
            except Exception:
                pass
            chop_s = "NA"
            try:
                c = float(last.get("chop"))
                if c == c:
                    chop_s = "%.0f" % c
            except Exception:
                pass
            try:
                ctx["market_state"] = "CTX:%s|D1:%s|H1:%s|M15:%s|CHOP:%s" % (
                    getattr(signal, "context_mode", "?"), getattr(signal, "d1_trend", "?"),
                    getattr(signal, "h1_trend", "?"), getattr(signal, "m15_trend", "?"), chop_s)
            except Exception:
                pass
        if ob_data:
            try:
                ctx["ob_vol"] = float(ob_data.get("vol"))
            except Exception:
                pass
            try:
                bid = float(ob_data.get("bid"))
                ask = float(ob_data.get("ask"))
                if bid > 0:
                    ctx["spread"] = (ask - bid) / bid * 1e4
            except Exception:
                pass
            try:
                sb = sum(float(x[1]) for x in ob_data.get("bids", []))
                sa = sum(float(x[1]) for x in ob_data.get("asks", []))
                if (sb + sa) > 0:
                    ctx["imbalance"] = (sb - sa) / (sb + sa)
            except Exception:
                pass
    except Exception as e:
        try:
            diag(symbol, "CTX: снимок не удался: %s" % e)
        except Exception:
            pass
    return ctx


CTX_KEYS = ("atr", "adx", "ema_dist", "structure",
            "market_state", "ob_vol", "spread", "imbalance")
# === /ENTRY_CONTEXT_PATCH_V34 ===

tracker = PerformanceTracker()

# ==========================================
# ⚙️ TERMINATOR BOT
# ==========================================
class FSM:
    @staticmethod
    def get_state(): return state_db.get('FSM', {}).get('state', 'SCAN')
    @staticmethod
    def get_target(): return state_db.get('FSM', {}).get('target')
    @staticmethod
    def set_state(state, target=None, reason=''):
        current = state_db.get('FSM', {})
        if current.get('state') == state and current.get('target') == target: return
        state_db['FSM'] = {'state': state, 'target': target, 'reason': reason, 'ts': time.time()}
        save_state(state_db)
    @staticmethod
    def recover(active_symbols):
        target = FSM.get_target()
        if target and target in active_symbols: FSM.set_state('POSITION_OPEN', target, 'RESTART')
        elif target and state_db.get(target, {}).get('pending_order'): FSM.set_state('WAIT_LIMIT', target, 'RESTART')
        elif active_symbols: FSM.set_state('POSITION_OPEN', active_symbols[0], 'RESTART')
        else: FSM.set_state('SCAN', None, 'RESTART')

class TerminatorBot:
    def __init__(self, symbol):
        self.symbol = symbol
        if self.symbol not in state_db:
            state_db[self.symbol] = {
                'is_breakeven_set': False, 'is_trailing_set': False, 'is_early_be_set': False,
                'state': 'SCAN', 'entered_notified': False, 'active_strategy': None,
                'last_signal': None, 'context_mode': 'FREE', 'risk_modifier': 1.0
            }
            save_state(state_db)
        self.db = state_db[self.symbol]
        self.qty_decimals, self.price_decimals = self.get_symbol_precision()

    @api_retry(max_retries=3)
    def get_symbol_precision(self):
        try:
            info = session.get_instruments_info(category="linear", symbol=self.symbol)
            if info.get('retCode') == 0:
                p_dec = max(0, int(-math.log10(float(info['result']['list'][0]['priceFilter']['tickSize']))))
                q_dec = max(0, int(-math.log10(float(info['result']['list'][0]['lotSizeFilter']['qtyStep']))))
                return q_dec, p_dec
            return 2, 4
        except Exception as e:
            log_msg(self.symbol, f"⚠ Ошибка получения точности: {e}", "ERROR")
            return 2, 4

    @api_retry(max_retries=3)
    def get_real_balance(self):
        try:
            res = session.get_wallet_balance(accountType="UNIFIED")
            if res.get('retCode') == 0:
                coins = res.get('result', {}).get('list', [{}])[0].get('coin', [])
                usdt_data = next((item for item in coins if item.get('coin') == 'USDT'), None)
                return float(usdt_data.get('walletBalance', 0)) if usdt_data else 0.0
            return 0.0
        except Exception as e:
            log_msg(self.symbol, f"⚠ Ошибка баланса: {e}", "ERROR")
            return 0.0

    @api_retry(max_retries=3)
    def execute_strategy_signal(self, signal, df_context=None):
        now = time.time()
        if now < self.db.get('cooldown_until', 0): return
        
        struct_event = ''
        if df_context is not None:
            struct = StructureDetector(self.symbol).analyze_structure(df_context)
            struct_event = struct.get('event', '')
        
        sig_hash = f'{self.symbol}_{signal.direction}_{round(signal.entry_price, 4)}_{signal.strategy_name}_{struct_event}'
        if sig_hash == self.db.get('last_executed_hash', ''): return
        
        # pending_order_id не очищается после сделки, поэтому проверять его
        # нельзя: при пропущенном входе уходило «Ордер принят» со старым ID.
        result = self._execute_strategy_signal_raw(signal)
        if result != "RETRY":
            self.db['last_executed_hash'] = sig_hash
        if result is True:
            if self.db.get('pending_order'):
                FSM.set_state('WAIT_LIMIT', self.symbol, 'ENTRY')
            else:
                FSM.set_state('POSITION_OPEN', self.symbol, 'MARKET_FILL')
            save_state(state_db)
            msg = f'✅ Ордер принят\nМонета: {self.symbol}\nНаправление: {signal.direction}\nРежим: {signal.context_mode} (Объем: {signal.risk_modifier}x)\nТренд D1: {signal.d1_trend} | H1: {signal.h1_trend}\nВход: {signal.entry_price} (ордер {self.db.get("order_price", signal.entry_price)} {self.db.get("order_tif", "")})\nSL: {signal.stop_loss}\nTP: {signal.take_profit}\nПричина: {signal.reason}\nID: `{self.db.get("pending_order_id", "UNKNOWN")}`'
            log_msg(self.symbol, msg, 'ENTRY')

    def _execute_strategy_signal_raw(self, signal: StrategySignal):
        if self.db.get('pending_order'):
            try:
                ord_res = session.get_open_orders(category="linear", symbol=self.symbol)
                if ord_res and ord_res.get('retCode') == 0:
                    if len(ord_res.get('result', {}).get('list', [])) == 0:
                        pos_res = session.get_positions(category="linear", symbol=self.symbol)
                        if pos_res and pos_res.get('retCode') == 0:
                            has_pos = any(float(p['size']) > 0 for p in pos_res.get('result', {}).get('list', []))
                            if not has_pos:
                                self.db['pending_order'] = False
                                save_state(state_db)
            except Exception as e:
                log_msg(self.symbol, f"⚠ Ошибка при проверке отложенного ордера: {e}", "ERROR")

        try:
            ord_res = session.get_open_orders(category="linear", symbol=self.symbol)
            if ord_res and ord_res.get('retCode') == 0 and len(ord_res.get('result', {}).get('list', [])) > 0: return
        except Exception: return

        try:
            pos_res = session.get_positions(category="linear", symbol=self.symbol)
            if pos_res and pos_res.get('retCode') == 0:
                for p in pos_res.get('result', {}).get('list', []):
                    if float(p['size']) > 0: return
        except Exception: return

        now = time.time()
        last_order_time = self.db.get('last_order_time', 0)
        if now - last_order_time < 300 and self.db.get('pending_order'): return
        if now - last_order_time < 5: return "RETRY"

        bal = self.get_real_balance()
        if bal < 5: return

        # FIX-5: доступную маржу читаем, но объём считаем от стопа (ниже).
        total_available = 0.0
        try:
            bal_res = session.get_wallet_balance(accountType="UNIFIED")
            if bal_res and bal_res.get('retCode') == 0:
                acc_list = bal_res.get('result', {}).get('list', [{}])
                if acc_list:
                    total_available = float(acc_list[0].get('totalAvailableBalance', 0))
        except Exception:
            return
        if total_available <= 0:
            return

        try: session.set_leverage(category="linear", symbol=self.symbol, buyLeverage=str(CONFIG["LEVERAGE"]), sellLeverage=str(CONFIG["LEVERAGE"]))
        except: pass

        self.db['last_order_time'] = time.time()
        save_state(state_db)

        p2 = round(signal.entry_price, self.price_decimals)
        sl = round(signal.stop_loss, self.price_decimals)
        tp = round(signal.take_profit, self.price_decimals)

        # ==================================================================
        # FIX-1: order_type читается, а не игнорируется.
        # Market-вход осмыслен только пока цена рядом с сигнальной, иначе
        # мы догоняем ушедшее движение. SL/TP пересчитываем от живой цены,
        # сохраняя исходный R:R.
        # ==================================================================
        desired_type = getattr(signal, "order_type", "Limit")
        if not CONFIG.get("RESPECT_ORDER_TYPE", True):
            desired_type = "Limit"
        strategy_type = desired_type
        maker_entry = CONFIG.get("MAKER_ENTRY", True)

        if desired_type == "Market" or maker_entry:
            try:
                tk = session.get_tickers(category="linear", symbol=self.symbol)
                t0 = tk["result"]["list"][0]
                live = float(t0["lastPrice"])
                bid, ask = float(t0["bid1Price"]), float(t0["ask1Price"])
            except Exception as e:
                log_msg(self.symbol, f"⚠ Не удалось получить тикер: {e}", "ERROR")
                return
            if bid <= 0 or ask <= 0:
                return

        new_entry = p2
        if desired_type == "Market":
            drift_bps = (live - p2) / p2 * 1e4
            if signal.direction == "Short":
                drift_bps = -drift_bps
            if drift_bps > CONFIG.get("MAX_MARKET_DRIFT_BPS", 12.0):
                diag(self.symbol, f"Market-вход пропущен — цена ушла на "
                                  f"{drift_bps:.1f}bps от сигнальной "
                                  f"(лимит {CONFIG.get('MAX_MARKET_DRIFT_BPS', 12.0):.0f}bps)")
                return
            new_entry = live
            if maker_entry:
                new_entry = bid if signal.direction == "Long" else ask
        elif maker_entry:
            # Лимитка выше bid (лонг) исполнялась сразу и платила taker.
            new_entry = min(p2, bid) if signal.direction == "Long" else max(p2, ask)

        if maker_entry:
            desired_type = "Limit"

        # Сдвигаем SL/TP вместе с входом: дистанция до стопа и R:R сигнала
        # сохраняются, на них опираются сайзинг, переанкоровка v26 и r_multiple.
        new_entry = round(new_entry, self.price_decimals)
        if new_entry != p2:
            shift = new_entry - p2
            sl = round(sl + shift, self.price_decimals)
            tp = round(tp + shift, self.price_decimals)
            p2 = new_entry

        risk_bps = abs(p2 - sl) / p2 * 1e4 if p2 > 0 else 0.0

        # ==================================================================
        # FIX-5: объём от ширины стопа, а не от доли депозита.
        #   qty = (баланс * риск%) / дистанция_до_стопа
        # Потолок по марже считается от ФАКТИЧЕСКИ доступных средств.
        # ==================================================================
        stop_dist = abs(p2 - sl)
        if stop_dist <= 0:
            return
        risk_usdt = bal * (CONFIG.get("RISK_PER_TRADE_PCT", 1.0) / 100.0) * signal.risk_modifier
        q_raw = risk_usdt / stop_dist

        max_notional = total_available * CONFIG["LEVERAGE"] * CONFIG.get("MAX_MARGIN_USAGE", 0.5)
        if q_raw * p2 > max_notional:
            q_raw = max_notional / p2
            diag(self.symbol, f"SIZING: объём урезан по марже до {q_raw * p2:.2f} USDT")

        q = round(q_raw, self.qty_decimals)
        notional = q * p2
        if q <= 0 or notional < CONFIG.get("MIN_NOTIONAL_USDT", 5.0):
            diag(self.symbol, f"SIZING: объём {notional:.2f} USDT ниже минимума "
                              f"{CONFIG.get('MIN_NOTIONAL_USDT', 5.0):.0f} USDT "
                              f"(стоп {risk_bps:.0f}bps, риск {risk_usdt:.2f} USDT)")
            return

        if q > 0:
            api_side = "Buy" if signal.direction == "Long" else "Sell"
            params = {
                "category": "linear", "symbol": self.symbol, "side": api_side,
                "orderType": desired_type, "qty": str(q),
                "stopLoss": str(sl), "takeProfit": str(tp), "positionIdx": 0
            }
            if desired_type == "Limit":
                params["price"] = str(p2)
                if maker_entry:
                    params["timeInForce"] = "PostOnly"
            else:
                params["timeInForce"] = "IOC"
            tif = params.get("timeInForce", "GTC")
            diag(self.symbol, f"ORDER {desired_type} {tif} {api_side} qty={q} "
                              f"notional={notional:.2f} entry={p2} sl={sl} tp={tp} "
                              f"| стоп {risk_bps:.0f}bps, риск {risk_usdt:.2f} USDT "
                              f"| сигнал {signal.strategy_name} {strategy_type} @ {signal.entry_price}")
            response = session.place_order(**params)
            
            if response and response.get("retCode") == 0:
                order_id = response.get("result", {}).get("orderId", "UNKNOWN")
                time.sleep(0.5)
                try:
                    # Market/IOC исполняется мгновенно и в open_orders не висит,
                    # поэтому для него подтверждение синхронизации не требуем.
                    if desired_type == "Market":
                        sync_ok = True
                    else:
                        status = self._order_status(order_id)
                        # None: статус неизвестен — ведём ордер, smart_pending_manager
                        # разберётся на следующем цикле; иначе он остался бы сиротой.
                        sync_ok = status is None or status in ('New', 'PartiallyFilled', 'Filled')
                        if not sync_ok:
                            diag(self.symbol, f"MAKER: ордер {order_id[:8]} не встал (status={status or 'нет'}) — "
                                              f"PostOnly пересёк бы стакан, повтор на следующем цикле")
                            return "RETRY"
                    if sync_ok:
                        self.db['pending_order'] = (desired_type == "Limit")
                        self.db['pending_order_id'] = order_id
                        self.db['order_price'] = p2
                        self.db['order_tif'] = tif
                        self.db['pending_ttl_sec'] = (CONFIG.get("MAKER_MARKET_TTL_SEC", 180)
                                                      if strategy_type == "Market"
                                                      else CONFIG.get("ORDER_TIMEOUT_MINUTES", 15) * 60)
                        self.db['is_breakeven_set'] = False
                        self.db['is_trailing_set'] = False
                        self.db['is_early_be_set'] = False
                        self.db['entered_notified'] = False
                        self.db['state'] = 'SNIPER'
                        self.db['active_strategy'] = signal.strategy_name
                        self.db['last_signal'] = signal.__dict__
                        # === ENTRY_CONTEXT_PATCH_V34 ===
                        self.db['entry_context'] = dict(getattr(self, '_entry_ctx', None) or {})
                        self.db['cancel_notified'] = False
                        try:
                            _d = mf_cache.fetch(self.symbol, "15", cache_ttl=180)
                            if _d is not None and not _d.empty:
                                self.db['last_struct_event'] = StructureDetector(self.symbol).analyze_structure(_d).get('event','')
                        except Exception:
                            pass
                        self.db['context_mode'] = signal.context_mode
                        self.db['risk_modifier'] = signal.risk_modifier
                        save_state(state_db)
                        return True
                except Exception as e:
                    log_msg(self.symbol, f"⚠ Ошибка при проверке статуса нового ордера: {e}", "ERROR")
            else:
                log_msg(self.symbol, f"⚠ Ошибка биржи\nКод: {response.get('retCode')}\nОписание: {response.get('retMsg')}", level="ERROR")

    def _order_status(self, order_id):
        """Статус ордера: сначала активные, затем история (исполненный
        или отменённый ордер из open_orders пропадает).
        '' — биржа ответила, ордера нет; None — биржа не ответила вовсе."""
        answered = False
        for fn in (session.get_open_orders, session.get_order_history):
            try:
                res = fn(category="linear", symbol=self.symbol, orderId=order_id)
                if res and res.get('retCode') == 0:
                    answered = True
                    lst = res.get('result', {}).get('list', [])
                    if lst:
                        return lst[0].get('orderStatus', '')
            except Exception:
                pass
        return '' if answered else None

    def smart_pending_manager(self, best_signal, ob_data, dfs=None):
        if not self.db.get('pending_order'): return
        pending_id = self.db.get('pending_order_id')
        if not pending_id: return
        
        try:
            status = self._order_status(pending_id)
            if status is None:
                return
            if status not in ('New', 'PartiallyFilled', 'Untriggered'):
                self.db['pending_order'] = False
                if status == 'Filled':
                    FSM.set_state('POSITION_OPEN', self.symbol, 'FILLED')
                    log_msg(self.symbol, f'✅ Ордер исполнен\nМонета: {self.symbol}\nID: `{pending_id}`', 'FILLED')
                else:
                    diag(self.symbol, f"PENDING {pending_id[:8]} снят биржей (status={status or 'нет'})")
                    FSM.set_state('SCAN', None, 'CANCELLED_EXT')
                save_state(state_db)
                return
        except Exception as e:
            log_msg(self.symbol, f"⚠ Ошибка при отмене/проверке Pending ордера: {e}", "ERROR")
            return
        
        last_sig_data = self.db.get('last_signal', {})
        order_price = last_sig_data.get('entry_price', 0)
        order_dir = last_sig_data.get('direction', '')
        now = time.time()
        score = 0
        factors = []
        
        if order_price > 0:
            dist = abs(ob_data.get('price', 0) - order_price) / order_price
            if dist > CONFIG.get('ORDER_CANCEL_DISTANCE', 0.01):
                score += 100; factors.append('Price Distance > 1.5%')
                
        if dfs and '15' in dfs and not dfs['15'].empty:
            df = dfs['15']
            struct = StructureDetector(self.symbol).analyze_structure(df)
            struct_evt = struct.get('event', '')
            if struct_evt in ['CHOCH_UP', 'CHOCH_DOWN', 'BOS_UP', 'BOS_DOWN']:
                is_against = False
                if order_dir == 'Long' and struct_evt in ['CHOCH_DOWN', 'BOS_DOWN']: is_against = True
                if order_dir == 'Short' and struct_evt in ['CHOCH_UP', 'BOS_UP']: is_against = True
                
                if is_against and struct_evt != self.db.get('last_struct_event', ''):
                    score += 100; factors.append(f'Trend Break Against Us ({struct_evt})')
                    self.db['cooldown_until'] = 0
            self.db['last_struct_event'] = struct_evt
        
        # === EXECUTION_FIX_PATCH_V17 ===
        last_order_time = self.db.get('last_order_time', 0)
        timeout_sec = self.db.get('pending_ttl_sec') or CONFIG.get("ORDER_TIMEOUT_MINUTES", 15) * 60
        if now - last_order_time > timeout_sec:
            score += 100
            factors.append(f'Timeout ({timeout_sec / 60:g}m)')
        
        age = int(now - self.db.get('last_order_time', now))
        diag(self.symbol, f"PENDING {pending_id[:8]} age={age}s score={score} "
             f"factors={factors} price={ob_data.get('price')} order={order_price} "
             f"struct={self.db.get('last_struct_event','')}")
        if score >= 100:
            diag(self.symbol, f"CANCEL {pending_id[:8]} after {age}s: {factors}")
            try:
                c_res = session.cancel_order(category='linear', symbol=self.symbol, orderId=pending_id)
                if c_res and c_res.get('retCode') == 0:
                    self.db['pending_order'] = False
                    self.db['cooldown_until'] = now + 900
                    save_state(state_db)
                    FSM.set_state('SCAN', None, 'CANCEL')
                    reason = ', '.join(factors)
                    log_msg(self.symbol, f'🗑 Ордер отменён\nМонета: {self.symbol}\nПричина: {reason}\nID: `{pending_id}`', 'CANCEL')
                else:
                    diag(self.symbol, f"cancel NOT confirmed: {c_res}")
            except Exception as e:
                diag(self.symbol, f"cancel EXCEPTION: {e}")
                log_msg(self.symbol, f"⚠ Ошибка при отмене ордера: {e}", "ERROR")

    @api_retry(max_retries=3)
    def manage_position(self, pos, df_m15, df_m5):
        entry_price = float(pos['avgPrice'])
        side = pos['side']
        size = float(pos['size'])
        mark_price = float(pos.get('markPrice', entry_price))

        if not self.db.get('entered_notified'):
            log_msg(self.symbol, f"✅ Ордер исполнен\nМонета: {self.symbol}\nНаправление: {side}\nЦена исполнения: {entry_price}", level="FILLED")
            self.db['entered_notified'] = True
            self.db['true_entry_time'] = time.time()
            save_state(state_db)
            signal_dict = self.db.get('last_signal', {})
            signal_obj = StrategySignal(**signal_dict) if signal_dict else None
            tracker.on_position_opened(self.symbol, entry_price, size, side, signal_obj)
            # === SL_TP_REANCHOR_PATCH_V26 ===
            try:
                if signal_dict and signal_dict.get('entry_price') and signal_dict.get('stop_loss'):
                    sig_entry = float(signal_dict['entry_price'])
                    sig_sl = float(signal_dict['stop_loss'])
                    sig_tp_raw = signal_dict.get('take_profit')
                    sig_tp = float(sig_tp_raw) if sig_tp_raw else None
                    risk_dist = abs(sig_entry - sig_sl)
                    reward_dist = abs(sig_tp - sig_entry) if sig_tp else None

                    if risk_dist > 0:
                        if side == "Buy":
                            anchored_sl = entry_price - risk_dist
                            anchored_tp = (entry_price + reward_dist) if reward_dist else None
                        else:
                            anchored_sl = entry_price + risk_dist
                            anchored_tp = (entry_price - reward_dist) if reward_dist else None

                        _, p_dec = self.get_symbol_precision()
                        anchored_sl_r = round(anchored_sl, p_dec)
                        anchored_tp_r = round(anchored_tp, p_dec) if anchored_tp else None
                        old_sl_r = round(sig_sl, p_dec)
                        old_tp_r = round(sig_tp, p_dec) if sig_tp else None

                        if anchored_sl_r != old_sl_r or (anchored_tp_r is not None and anchored_tp_r != old_tp_r):
                            api_params = {"category": "linear", "symbol": self.symbol,
                                          "stopLoss": str(anchored_sl_r), "positionIdx": 0}
                            if anchored_tp_r is not None:
                                api_params["takeProfit"] = str(anchored_tp_r)
                            res = session.set_trading_stop(**api_params)
                            if res and res.get('retCode') == 0:
                                tp_line = f"TP: {old_tp_r} -> {anchored_tp_r}\n" if anchored_tp_r is not None else ""
                                log_msg(self.symbol,
                                    f"🎯 SL/TP ПЕРЕАНКОРЕНЫ\nМонета: {self.symbol}\n"
                                    f"Цена сигнала: {sig_entry} -> Реальный вход: {entry_price}\n"
                                    f"SL: {old_sl_r} -> {anchored_sl_r}\n{tp_line}"
                                    f"Риск в USD теперь точен относительно реального входа.")
                            else:
                                err = res.get('retMsg') if res else 'no response'
                                log_msg(self.symbol, f"⚠ Не удалось переанкорить SL/TP: {err}", "ERROR")
            except Exception as e:
                log_msg(self.symbol, f"⚠ Ошибка переанкоровки SL/TP: {e}", "ERROR")

        elif self.symbol not in tracker.active_trades:
            # === RESTART_TRACKING_RECOVERY_PATCH ===
            # entered_notified уже стоял True (позиция была открыта ДО перезапуска процесса),
            # но tracker.active_trades живёт только в памяти и сбрасывается при каждом старте.
            # Без этого блока закрытие такой позиции никогда не попадёт ни в Telegram,
            # ни в trades.db - check_closed_positions() просто не знает, что её нужно
            # отслеживать. Регистрируем позицию заново, чтобы её закрытие не потерялось.
            signal_dict = self.db.get('last_signal', {})
            signal_obj = StrategySignal(**signal_dict) if signal_dict else None
            tracker.on_position_opened(self.symbol, entry_price, size, side, signal_obj)

            
            # ИСПРАВЛЕНИЕ 3: Полностью удален блок отправки биржевого трейлинга Bybit при входе
            # Теперь AdaptivePositionManager - единственный и полноправный хозяин выхода.

        current_sl = float(pos.get('stopLoss', 0) or 0)
        current_mode = self.db.get('context_mode', 'FREE')
        
        initial_sl = self.db.get('last_signal', {}).get('stop_loss', current_sl)

        # === TRUE_RISK_DISTANCE_FIX_PATCH_V24 ===
        _sig = self.db.get('last_signal', {})
        original_risk_distance = None
        if _sig.get('entry_price') and _sig.get('stop_loss'):
            original_risk_distance = abs(float(_sig['entry_price']) - float(_sig['stop_loss']))

        apm = AdaptivePositionManager(
            df_m15=df_m15, df_m5=df_m5,
            entry_price=entry_price, current_price=mark_price,
            current_sl=current_sl, initial_sl=initial_sl, side=side,
            is_early_be_set=self.db.get('is_early_be_set', False),
            symbol=self.symbol, mode=current_mode,
            true_entry_time=self.db.get('true_entry_time'),
            original_risk_distance=original_risk_distance
        )

        action, proposed_sl, msg, current_r, atr_val, reason_str = apm.analyze()

        # === MFE_MAE_WIRING_PATCH_V31 ===
        try:
            tracker.update_excursion(self.symbol, current_r)
        except Exception as _exc_err:
            log_msg(self.symbol, f"[EXCURSION] ошибка трекинга: {_exc_err}", "ERROR")
        
        if action == 'TIME_EXIT':
            try:
                pos_check = session.get_positions(category="linear", symbol=self.symbol)
                if pos_check and pos_check.get('retCode') == 0:
                    pos_list = pos_check.get('result', {}).get('list', [])
                    if not pos_list or float(pos_list[0].get('size', 0)) <= 0:
                        return
            except Exception as e:
                log_msg(self.symbol, f"⚠ Ошибка при проверке закрытия позиции: {e}", "ERROR")

            try:
                session.cancel_all_orders(category="linear", symbol=self.symbol)
            except Exception as e:
                log_msg(self.symbol, f"⚠ Ошибка при отмене ордеров (Structural Exit): {e}", "ERROR")

            res = session.place_order(category="linear", symbol=self.symbol, side="Sell" if side=="Buy" else "Buy", orderType="Market", qty=str(size), reduceOnly=True)
            if res and res.get('retCode') == 0:
                log_msg(self.symbol, msg, 'CANCEL')
            return

        if action in ['SET_EARLY_BE', 'TRAIL'] and proposed_sl != current_sl:
            was_trailing_already_set = self.db.get('is_trailing_set', False)
            clear_tp_now = (action == 'TRAIL' and not was_trailing_already_set)

            if side == "Buy":
                final_sl = max(current_sl, proposed_sl) if current_sl > 0 else proposed_sl
            else:
                final_sl = min(current_sl, proposed_sl) if current_sl > 0 else proposed_sl
            
            min_dist = atr_val * 0.1
            if side == "Buy" and final_sl > mark_price - min_dist:
                final_sl = mark_price - min_dist
            elif side == "Sell" and final_sl < mark_price + min_dist:
                final_sl = mark_price + min_dist

            q_dec, p_dec = self.get_symbol_precision()
            final_sl_rounded = round(final_sl, p_dec)
            current_sl_rounded = round(current_sl, p_dec)

            if final_sl_rounded != current_sl_rounded:
                api_params = {"category": "linear", "symbol": self.symbol, "stopLoss": str(final_sl_rounded), "positionIdx": 0}
                if clear_tp_now:
                    api_params["takeProfit"] = "0"
                res = session.set_trading_stop(**api_params)
                if res and res.get('retCode') == 0:
                    tp_note = "\nTP снят с биржи — трейлинг теперь ведёт сделку без потолка" if clear_tp_now else ""
                    log_text = (
                        f"🛡 SL UPDATE\n"
                        f"Symbol: {self.symbol}\n"
                        f"Side: {side}\n"
                        f"Entry: {entry_price}\n"
                        f"Initial SL: {initial_sl}\n"
                        f"Old SL: {current_sl_rounded}\n"
                        f"New SL: {final_sl_rounded}\n"
                        f"R: {current_r:.2f}R\n"
                        f"ATR: {atr_val:.4f}\n"
                        f"Reason: {reason_str}"
                        f"{tp_note}"
                    )
                    log_msg(self.symbol, log_text)
                    
                    if action == 'SET_EARLY_BE': self.db['is_early_be_set'] = True
                    elif action == 'TRAIL': self.db['is_trailing_set'] = True
                    save_state(state_db)

@api_retry(max_retries=3)
def get_active_positions():
    res = session.get_positions(category="linear", settleCoin="USDT")
    if res and res.get('retCode') == 0:
        return [p for p in res['result']['list'] if float(p['size']) > 0]
    return []

# ==========================================
# 🚀 ГЛАВНЫЙ ЦИКЛ (MAIN)
# ==========================================

# === SCAN_DIAGNOSTICS_PATCH_V5 ===
def debug_scan_diagnostics(ticker, dfs, ob_data, has_signal):
    try:
        df_m15 = dfs.get("15")
        df_h1 = dfs.get("60")
        df_d1 = dfs.get("D")
        if df_m15 is None or not ob_data:
            return
        chop_val = df_m15['chop'].iloc[-1]
        atr_m15 = df_m15['atr'].iloc[-1]
        atr_pct = (atr_m15 / df_m15['close'].iloc[-1]) * 100
        sniper_dir = ob_data.get('dir')
        wall_score = ob_data.get('sniper_score', 0)

        ltp_reason = "wall_score=0 (net trigger)"
        if wall_score != 0:
            spread_pct = (ob_data['ask'] - ob_data['bid']) / ob_data['bid'] * 100
            ema_ok, atr_exh_ok, spread_ok = True, True, True
            if df_h1 is not None and not df_h1.empty:
                ema200_h1 = df_h1['ema_200'].iloc[-1]
                close_h1 = df_h1['close'].iloc[-1]
                if sniper_dir == "Long" and close_h1 < ema200_h1: ema_ok = False
                if sniper_dir == "Short" and close_h1 > ema200_h1: ema_ok = False
            if df_d1 is not None and not df_d1.empty:
                daily = df_d1.iloc[-1]
                travel = abs(df_m15['close'].iloc[-1] - daily['open'])
                limit = daily['atr'] * CONFIG["MAX_DAILY_ATR_EXHAUSTION"]
                if travel > limit: atr_exh_ok = False
            spread_ok = spread_pct <= CONFIG["MAX_SPREAD_PCT"]
            ltp_reason = f"EMA200H1={'OK' if ema_ok else 'FAIL'} ATR_EXH={'OK' if atr_exh_ok else 'FAIL'} SPREAD={'OK' if spread_ok else 'FAIL'}"

        smc_reason = "n/a"
        try:
            sr = SREngine()
            raw_d1 = sr.detect_trend_breaks(df_d1, 3, 3) if df_d1 is not None else []
            raw_h1 = sr.detect_trend_breaks(df_h1, 5, 5) if df_h1 is not None else []
            zones = sr.cluster_and_merge_levels(df_h1, raw_d1 + raw_h1) if (df_h1 is not None and (raw_d1 or raw_h1)) else []
            cur_price = df_m15['close'].iloc[-1]
            closest, mind = None, float('inf')
            for z in zones:
                d = abs(cur_price - z['price'])
                if d < mind: mind, closest = d, z
            if not closest or mind > atr_m15 * 3:
                smc_reason = "no S/R zone in range"
            else:
                fb = FalseBreakoutEngine(df_m15, closest['price'], closest['type'])
                b1, b2 = fb.check_1bar_false_breakout(), fb.check_2bar_false_breakout()
                if not (b1['detected'] or b2['detected']):
                    smc_reason = f"near {closest['type']}, no false-breakout pattern yet"
                else:
                    is_choppy = chop_val > 61.8 if not pd.isna(chop_val) else False
                    vol_surge = df_m15['volume'].iloc[-1] > (df_m15['vol_sma'].iloc[-1] * 1.15)
                    if is_choppy:
                        smc_reason = "false-breakout OK but CHOP too high (flat market)"
                    elif not vol_surge:
                        smc_reason = "false-breakout OK but no volume surge"
                    else:
                        smc_reason = "false-breakout+chop+volume OK -> check EMA50/RR/candle filters"
        except Exception as e:
            smc_reason = f"calc error: {e}"

        mom_reason = "n/a"
        try:
            lookback = CONFIG["MOMENTUM_LOOKBACK"]
            if df_m15 is not None and len(df_m15) >= lookback + 2:
                prior = df_m15.iloc[-(lookback + 1):-1]
                range_high = prior['high'].max()
                range_low = prior['low'].min()
                last = df_m15.iloc[-1]
                candle_range = last['high'] - last['low']
                if candle_range <= 0:
                    mom_reason = "invalid candle"
                else:
                    body_ratio = abs(last['close'] - last['open']) / candle_range
                    direction, wick_ratio = None, 1.0
                    if last['close'] > range_high and last['close'] > last['open']:
                        direction = "Long"; wick_ratio = (last['high'] - last['close']) / candle_range
                    elif last['close'] < range_low and last['close'] < last['open']:
                        direction = "Short"; wick_ratio = (last['close'] - last['low']) / candle_range
                    if direction is None:
                        mom_reason = f"inside {lookback}-bar range (H={range_high:.4g} L={range_low:.4g})"
                    elif body_ratio < CONFIG["MOMENTUM_BODY_RATIO"] or wick_ratio > CONFIG["MOMENTUM_MAX_WICK_RATIO"]:
                        mom_reason = f"{direction} breakout but weak candle (body={body_ratio:.2f} wick={wick_ratio:.2f})"
                    else:
                        vol_sma_last = df_m15['vol_sma'].iloc[-1]
                        vol_ratio = (last['volume'] / vol_sma_last) if vol_sma_last else 0
                        if vol_ratio < CONFIG["MOMENTUM_VOL_MULT"]:
                            mom_reason = f"{direction} breakout+candle OK but vol={vol_ratio:.2f}x (need {CONFIG['MOMENTUM_VOL_MULT']}x)"
                        else:
                            chop_v = df_m15['chop'].iloc[-1]
                            if not pd.isna(chop_v) and chop_v > 61.8:
                                mom_reason = f"{direction} breakout+vol OK but CHOP too high"
                            else:
                                ema200_h1 = df_h1['ema_200'].iloc[-1] if (df_h1 is not None and not df_h1.empty) else None
                                close_h1 = df_h1['close'].iloc[-1] if (df_h1 is not None and not df_h1.empty) else None
                                if ema200_h1 is not None:
                                    if direction == "Long" and close_h1 < ema200_h1:
                                        mom_reason = f"{direction} ALL OK but against H1 trend"
                                    elif direction == "Short" and close_h1 > ema200_h1:
                                        mom_reason = f"{direction} ALL OK but against H1 trend"
                                    else:
                                        mom_reason = f"{direction} ALL CONDITIONS MET"
                                else:
                                    mom_reason = f"{direction} candle+vol+chop OK, no H1 data"
        except Exception as e:
            mom_reason = f"calc error: {e}"

        status = "SIGNAL_FOUND" if has_signal else "no_signal"
        print(f"[SCAN] {ticker:10} | chop={chop_val:5.1f} ATR%={atr_pct:.3f} | wall={wall_score} dir={sniper_dir} | "
              f"LTP:[{ltp_reason}] | SMC:[{smc_reason}] | MOM:[{mom_reason}] | {status}")
    except Exception as e:
        print(f"[SCAN] {ticker} diagnostics error: {e}")

if __name__ == "__main__":
    # === DEAD_TICKER_REMOVAL_PATCH_V32 ===
    # MATICUSDT: status=Closed на Bybit с сентября 2024 (миграция MATIC->POL).
    # POLUSDT торгуется, но НЕ добавлен сознательно — новый инструмент
    # посреди набора выборки сломал бы сравнение версий.
    COIN_LIST = ["XRPUSDT", "DOTUSDT", "ARBUSDT", "LINKUSDT", "AVAXUSDT", "SOLUSDT", "ADAUSDT"]

    _stale = [k for k, v in state_db.items()
              if k.endswith("USDT") and k not in COIN_LIST and isinstance(v, dict)
              and not v.get('pending_order')]
    for k in _stale:
        del state_db[k]
    if _stale:
        save_state(state_db)
        diag("STARTUP", f"удалены записи монет вне COIN_LIST: {_stale}")

    start_msg = (f"===================================\n"
                 f"🚀 PREDATOR v35\n"
                 f"🤖 Бот запущен.\n"
                 f"Монет в работе: {len(COIN_LIST)}\n"
                 f"Плечо: {CONFIG['LEVERAGE']}x | "
                 f"Риск на сделку: {CONFIG['RISK_PER_TRADE_PCT']}%\n"
                 f"Стратегий: 3 (TrendPRO, SMC, Momentum)\n"
                 f"Вход: {'maker-only (PostOnly)' if CONFIG.get('MAKER_ENTRY', True) else 'v34 (Market/Limit)'}\n"
                 f"===================================")
    print(start_msg)
    notify.send('STARTUP', 'START', start_msg) 

    mf_cache = MultiTimeframeCache(session)
    strategy_manager = StrategyManager()

    while True:
        try:
            active_positions = get_active_positions()
            active_syms = [p['symbol'] for p in active_positions] if active_positions else []
            FSM.recover(active_syms)
            
            best_global_signal = None
            best_global_bot = None
            best_global_df = None
            
            time.sleep(3)

            active_pos = get_active_positions()
            active_symbols = [p['symbol'] for p in active_pos] if active_pos else []
            tracker.check_closed_positions(active_symbols, session)

            if active_pos:
                for pos in active_pos:
                    bot = TerminatorBot(pos['symbol'])
                    df_m15 = mf_cache.fetch(pos['symbol'], "15", cache_ttl=180)
                    df_m5 = mf_cache.fetch(pos['symbol'], "5", cache_ttl=60)
                    if df_m15 is not None and df_m5 is not None:
                        bot.manage_position(pos, df_m15, df_m5)

            target_list = [FSM.get_target()] if FSM.get_state() != 'SCAN' and FSM.get_target() else COIN_LIST
            for ticker in target_list:
                if ticker in active_symbols:
                    continue

                try:
                    bot = TerminatorBot(ticker)

                    df_h4 = mf_cache.fetch(ticker, "240", cache_ttl=3600)
                    df_h1 = mf_cache.fetch(ticker, "60", cache_ttl=900)
                    df_m15 = mf_cache.fetch(ticker, "15", cache_ttl=180)
                    df_m5 = mf_cache.fetch(ticker, "5", cache_ttl=60)
                    df_1d = mf_cache.fetch(ticker, "D", cache_ttl=14400) 

                    if df_h4 is None or df_h1 is None or df_m15 is None or df_m5 is None or df_1d is None: continue
                    dfs = {"240": df_h4, "60": df_h1, "15": df_m15, "5": df_m5, "D": df_1d}

                    res = session.get_orderbook(category="linear", symbol=ticker, limit=CONFIG["OB_DEPTH"])
                    if not res or res.get('retCode') != 0: continue

                    bids, asks = res['result']['b'], res['result']['a']
                    if not bids or not asks: continue

                    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
                    sum_b_vol = sum(float(l[1]) for l in bids)
                    sum_a_vol = sum(float(l[1]) for l in asks)

                    wall_b = max(bids, key=lambda x: float(x[1]))
                    wall_a = max(asks, key=lambda x: float(x[1]))

                    sniper_dir = "Long" if float(wall_b[1]) > float(wall_a[1]) else "Short"
                    best_wall_p = float(wall_b[0]) if sniper_dir == "Long" else float(wall_a[0])
                    best_wall_v = float(wall_b[1]) if sniper_dir == "Long" else float(wall_a[1])

                    avg_vol = (sum_b_vol if sniper_dir=="Long" else sum_a_vol) / len(bids) + 1e-8
                    sniper_score = 100 if (best_wall_v / avg_vol) >= CONFIG["WALL_MULTIPLIER"] else 0

                    ob_data = {
                        'bids': bids, 'asks': asks, 'dir': sniper_dir,
                        'price': best_wall_p, 'vol': best_wall_v,
                        'ask': best_ask, 'bid': best_bid, 'sniper_score': sniper_score
                    }

                    winrate = tracker.get_coin_stats(ticker) or 50.0

                    best_signal = strategy_manager.evaluate_all(ticker, dfs, ob_data, winrate)
                    # === ENTRY_CONTEXT_PATCH_V34 ===
                    if best_signal:
                        bot._entry_ctx = snapshot_entry_context(ticker, best_signal, dfs, ob_data)
                        _f = sum(1 for _v in bot._entry_ctx.values() if _v is not None)
                        diag(ticker, "CTX %d/8 atr=%s adx=%s ema_dist=%s struct=%s "
                                     "state=%s ob_vol=%s spread=%s imb=%s" % (
                             _f, bot._entry_ctx['atr'], bot._entry_ctx['adx'],
                             bot._entry_ctx['ema_dist'], bot._entry_ctx['structure'],
                             bot._entry_ctx['market_state'], bot._entry_ctx['ob_vol'],
                             bot._entry_ctx['spread'], bot._entry_ctx['imbalance']))
                    debug_scan_diagnostics(ticker, dfs, ob_data, best_signal is not None)

                    bot.smart_pending_manager(best_signal, ob_data, dfs=dfs)

                    if best_signal:
                        if FSM.get_state() == 'SCAN':
                            if not best_global_signal or best_signal.get_final_score() > best_global_signal.get_final_score():
                                best_global_signal = best_signal
                                best_global_bot = bot
                                best_global_df = dfs.get('15')

                    time.sleep(0.5) 
                except Exception as e:
                    log_msg(ticker, f"⚠ Локальная ошибка парсинга монеты: {e}", "ERROR")

            if FSM.get_state() == 'SCAN' and best_global_signal and best_global_bot:
                best_global_bot.execute_strategy_signal(best_global_signal, best_global_df)
                
            time.sleep(5)
        except Exception as e:
            log_msg("GLOBAL", f"⚠ Критическая ошибка в главном цикле (FSM/API): {e}", "ERROR")
            time.sleep(10)
