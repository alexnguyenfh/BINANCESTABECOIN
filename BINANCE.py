# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
from stable_baselines3 import PPO

from binance.um_futures import UMFutures
from binance.client import Client
import telepot

# =========================
# CẤU HÌNH CHUNG
# =========================
# Telegram
token = '' # telegram token
receiver_id = 685322834 # https://api.telegram.org/bot<TOKEN>/getUpdates
bot = telepot.Bot(token)

# Phí 0.05% mở + 0.05% đóng (áp dụng trong ENV RL để tính reward)
FEE_RATE = 0.0005
# ATR
ATR_LEN = 14
ATR_SL_MULT = 1.5   # SL = 1.5 x ATR
RR_MULT = 2.0       # TP = 2.0 x ATR
# Làm tròn theo tick; BNBUSDT thường 0.1 -> 1 chữ số
TICK_ROUND_DEC = 1

# =========================
# HÀM TIỆN ÍCH
# =========================
def round_tick(px: float, decimals: int = TICK_ROUND_DEC) -> float:
    """Làm tròn giá theo số chữ số thập phân (tick)."""
    return float(f"{px:.{decimals}f}")

def nudge_sl_tp(entry: float, sl: float, tp: float, side: str):
    """
    Tránh lỗi "stop price would trigger immediately".
    side: "LONG" hoặc "SHORT"
    """
    if side == "LONG":
        # SL phải < entry, TP phải > entry
        if sl >= entry:
            sl = entry * 0.999
        if tp <= entry:
            tp = entry * 1.001
    else:
        # SHORT: SL phải > entry, TP phải < entry
        if sl <= entry:
            sl = entry * 1.001
        if tp >= entry:
            tp = entry * 0.999
    return sl, tp

# =========================
# KẾT NỐI BINANCE
# =========================
def make_binance_client():
    api_key = os.getenv("BINANCE_KEY", "")
    secret_key = os.getenv("BINANCE_SECRET", "gPiSoULgHIf0hTANHEkdY9hMrX8Ic2DiBHedF3afjRjaxTzMjmjFHyYhpFX0zt9G")
    client = Client(api_key=api_key, api_secret=secret_key, tld="com")


client = make_binance_client()

def is_position_open(symbol):
    """Kiểm tra xem đang có vị thế mở cho symbol hay không (hedge mode)."""
    try:
        positions = client.futures_account()['positions']
        for pos in positions:
            if pos['symbol'] == symbol and abs(float(pos['positionAmt'])) > 0:
                return True
        return False
    except Exception as e:
        print(f"Lỗi is_position_open: {e}")
        return False

def cancelallorder(symbol):
    """Xoá toàn bộ SL/TP đang mở (STOP_MARKET / TAKE_PROFIT_MARKET)."""
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            if order.get('type') in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                print(f"Canceled order: {order['orderId']} ({order['type']})")
    except Exception as e:
        print(f"Lỗi cancelallorder: {e}")

# =========================
# ĐẶT LỆNH VỚI ATR (SL/TP)
# =========================
def molong(loaicoin: str, soluong: float, atr_value: float):
    """
    Mở LONG, đặt SL = entry - 1.5*ATR, TP = entry + 2*ATR
    Dùng STOP_MARKET / TAKE_PROFIT_MARKET.
    """
    try:
        px_now = float(client.futures_symbol_ticker(symbol=loaicoin)['price'])
        px_now = round_tick(px_now, TICK_ROUND_DEC)

        sl = px_now - ATR_SL_MULT * atr_value
        tp = px_now + RR_MULT * atr_value
        sl, tp = nudge_sl_tp(px_now, sl, tp, side="LONG")
        sl = round_tick(sl, TICK_ROUND_DEC)
        tp = round_tick(tp, TICK_ROUND_DEC)

        # Mở LONG
        client.futures_create_order(
            symbol=loaicoin, side="BUY", type="MARKET",
            quantity=soluong, positionSide="LONG"
        )
        # Đặt SL
        client.futures_create_order(
            symbol=loaicoin, side="SELL", type="STOP_MARKET",
            stopPrice=sl, quantity=soluong, positionSide="LONG"
        )
        # Đặt TP
        client.futures_create_order(
            symbol=loaicoin, side="SELL", type="TAKE_PROFIT_MARKET",
            stopPrice=tp, quantity=soluong, positionSide="LONG"
        )
        print(f"Opened LONG {loaicoin} @ {px_now}, SL={sl}, TP={tp}")
    except Exception as e:
        print(f"Lỗi molong: {e}")

def donglong(loaicoin: str, soluong: float):
    """Đóng LONG bằng lệnh MARKET."""
    try:
        client.futures_create_order(symbol=loaicoin, side="SELL", type="MARKET",
                                    quantity=soluong, positionSide="LONG")
        print("Closed LONG")
    except Exception as e:
        print(f"Lỗi donglong: {e}")

def moshort(loaicoin: str, soluong: float, atr_value: float):
    """
    Mở SHORT, SL = entry + 1.5*ATR, TP = entry - 2*ATR
    """
    try:
        px_now = float(client.futures_symbol_ticker(symbol=loaicoin)['price'])
        px_now = round_tick(px_now, TICK_ROUND_DEC)

        sl = px_now + ATR_SL_MULT * atr_value
        tp = px_now - RR_MULT * atr_value
        sl, tp = nudge_sl_tp(px_now, sl, tp, side="SHORT")
        sl = round_tick(sl, TICK_ROUND_DEC)
        tp = round_tick(tp, TICK_ROUND_DEC)

        client.futures_create_order(
            symbol=loaicoin, side="SELL", type="MARKET",
            quantity=soluong, positionSide="SHORT"
        )
        client.futures_create_order(
            symbol=loaicoin, side="BUY", type="STOP_MARKET",
            stopPrice=sl, quantity=soluong, positionSide="SHORT"
        )
        client.futures_create_order(
            symbol=loaicoin, side="BUY", type="TAKE_PROFIT_MARKET",
            stopPrice=tp, quantity=soluong, positionSide="SHORT"
        )
        print(f"Opened SHORT {loaicoin} @ {px_now}, SL={sl}, TP={tp}")
    except Exception as e:
        print(f"Lỗi moshort: {e}")

def dongshort(loaicoin: str, soluong: float):
    """Đóng SHORT bằng lệnh MARKET."""
    try:
        client.futures_create_order(symbol=loaicoin, side="BUY", type="MARKET",
                                    quantity=soluong, positionSide="SHORT")
        print("Closed SHORT")
    except Exception as e:
        print(f"Lỗi dongshort: {e}")

# =========================
# DATA & INDICATORS
# =========================
def get_data(symbol: str = "BNBUSDT", interval: str = "5m", limit: int = 1500) -> pd.DataFrame:
    clientfutu = UMFutures()
    candles = clientfutu.klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(candles, columns=[
        "Time", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteAssetVolume", "NumberOfTrades",
        "TakerBuyBaseVolume", "TakerBuyQuoteVolume", "Ignore"
    ])

    df = df[["Time", "Open", "High", "Low", "Close", "Volume"]]
    df["Time"] = pd.to_datetime(df["Time"], unit="ms")
    df.set_index("Time", inplace=True)
    df = df.astype(float)

    # Indicators
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=1, multiplier=1.0)
    df['supertrend_direction'] = supertrend['SUPERTd_1_1.0']
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['adx'] = adx['ADX_14']
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['Closekecuoi'] = df['Close'].shift(1)
    df['xuhuonggia'] = df['Close'] - df['Closekecuoi']

    macd_result = ta.macd(df['Close'])
    df['macd1'] = macd_result['MACDh_12_26_9']
    df['macd2'] = macd_result['MACDh_12_26_9'].shift(1)
    df['xuhuongmacd'] = df['macd1'] - df['macd2']

    bollinger = ta.bbands(df['Close'], length=20, std=2)
    df['bollinger_mid'] = bollinger['BBM_20_2.0']
    df['bollinger_upper'] = bollinger['BBU_20_2.0']
    df['bollinger_lower'] = bollinger['BBL_20_2.0']
    df['hieusoboll'] = df['bollinger_upper'] - df['bollinger_lower']

    df['obv'] = ta.obv(df['Close'], df['Volume'])

    # ATR cho giao dịch thực (lấy tại điểm cuối)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=ATR_LEN)

    df = df.dropna()
    return df

# =========================
# ENV PPO (tích phí giao dịch)
# =========================
class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 1000.0  # Vốn ban đầu
        self.position = 0      # 0: flat, 1: đang giữ (long 1 đơn vị)
        self.entry_price = 0.0
        self.done = False
        self.reward = 0.0
        self.history = []  # [(step, price, action, ...), ...]

        # Số feature dùng cho RL (đúng theo các cột tính toán)
        self.n_features = 13
        self.action_space = gym.spaces.Discrete(3)  # 0: Mua, 1: Bán, 2: Giữ
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.n_features,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 1000.0
        self.position = 0
        self.entry_price = 0.0
        self.done = False
        self.reward = 0.0
        self.history = []
        obs = self._next_observation()
        return obs, {}

    def _next_observation(self):
        return self.df.iloc[self.current_step].values[:self.n_features]

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0.0

        # Hành động
        if action == 0:  # Mua (mở vị thế nếu đang flat)
            if self.position == 0:
                open_fee = current_price * FEE_RATE   # phí mở 0.05%
                self.balance -= open_fee
                self.position = 1
                self.entry_price = current_price
                self.history.append((self.current_step, current_price, 'Buy',
                                     f'fee_open={open_fee:.6f}'))

        elif action == 1:  # Bán (đóng vị thế nếu đang hold)
            if self.position == 1:
                gross = current_price - self.entry_price
                close_fee = current_price * FEE_RATE  # phí đóng 0.05%
                net = gross - close_fee
                reward = net
                self.balance += reward
                self.position = 0
                self.history.append((self.current_step, current_price, 'Sell',
                                     f'gross={gross:.6f}',
                                     f'fee_close={close_fee:.6f}',
                                     f'net={net:.6f}'))
                self.entry_price = 0.0

        # Cập nhật
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        obs = self._next_observation()
        return obs, reward, self.done, False, {}

# =========================
# VÒNG LẶP CHÍNH
# =========================
if __name__ == "__main__":
    SYMBOL = "BNBUSDT"
    QTY = 0.02

    while True:
        # (1) Tạo client mới mỗi vòng (tuỳ ý, có thể reuse)
        client = make_binance_client()

        # (2) Lấy dữ liệu + train PPO (như code gốc)
        df = get_data(SYMBOL)
        required_cols = [
            'supertrend_direction', 'hieusoboll', 'rsi',
            'bollinger_upper', 'bollinger_lower', 'bollinger_mid',
            'macd1', 'macd2', 'xuhuongmacd', 'obv', 'adx', 'atr'
        ]
        assert all(c in df.columns for c in required_cols), "Missing required features in df"

        env = TradingEnv(df)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)

        # (3) Kiểm tra mô hình
        obs, _ = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            if dones or truncated:
                break

        # (4) Lưu/Load model (giữ nguyên như bạn đang làm)
        model.save("ppo_BB.zip")
        print("Final Balance:", env.balance)
        print("Trading History:")
        for trade in env.history:
            print(trade)

        time.sleep(1)
        model = PPO.load("ppo_BB.zip")

        # (5) Lấy quyết định cuối cùng để gửi Telegram
        obs, _ = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            if dones or truncated:
                break

        if action == 0:
            hanhdong = "BUY"
        elif action == 1:
            hanhdong = "SELL"
        else:
            hanhdong = "HOLD"

        print("--------")
        print(hanhdong)
        try:
            bot.sendMessage(TELEGRAM_CHAT_ID, "--------")
            bot.sendMessage(TELEGRAM_CHAT_ID, str(hanhdong))
        except Exception as e:
            print(f"Lỗi gửi Telegram: {e}")

        # (6) In thông tin nhanh + quyết định giao dịch thực tế
        print("--------")
        print("supertrend_direction:", df['supertrend_direction'].iloc[-1])
        print("xuhuongmacd:", df['xuhuongmacd'].iloc[-1])
        vithedangmo = is_position_open(SYMBOL)
        print("vithedangmo:", vithedangmo)
        if df['Close'].iloc[-1] - df['Open'].iloc[-1] > 0:
            print("nen xanh")
        elif df['Close'].iloc[-1] - df['Open'].iloc[-1] < 0:
            print("nen do")
        print("Close:", df['Close'].iloc[-1])

        # Dữ liệu khung 1h để filter thêm (giữ như code gốc)
        try:
            clientfutu2 = UMFutures()
            candles2 = clientfutu2.klines(symbol=SYMBOL, interval='1h', limit=100)
            df2 = pd.DataFrame(candles2, columns=[
                "Time", "Open", "High", "Low", "Close", "Volume",
                "CloseTime", "QuoteAssetVolume", "NumberOfTrades",
                "TakerBuyBaseVolume", "TakerBuyQuoteVolume", "Ignore"
            ])
            df2 = df2[["Time", "Open", "High", "Low", "Close", "Volume"]]
            df2["Time"] = pd.to_datetime(df2["Time"], unit="ms")
            df2.set_index("Time", inplace=True)
            df2 = df2.astype(float)
        except Exception as e:
            print(f"Lỗi tải 1h data: {e}")
            df2 = None

        # (7) GIAO DỊCH THỰC: dùng ATR ở cây 5m cuối cùng
        atr_latest = float(df['atr'].iloc[-1])

        # Điều kiện của bạn (giữ nguyên), nhưng SL/TP đã chuyển qua ATR trong molong/moshort
        if env.balance > 1000:
            try:
                # Ví dụ điều kiện mở LONG (giữ như code gốc của bạn)
                if (not vithedangmo) and action == 0 and (df['xuhuongmacd'].iloc[-1] > 0) \
                   and (df['Close'].iloc[-1] - df['Open'].iloc[-1] > 0) \
                   and (df2 is not None and (df2['Close'].iloc[-1] - df2['Open'].iloc[-1] > 0)):
                    try:
                        cancelallorder(SYMBOL)
                    except Exception:
                        print("Không thể xoá lệnh cũ")

                    try:
                        molong(SYMBOL, QTY, atr_latest)
                        bot.sendMessage(TELEGRAM_CHAT_ID, "molong")
                    except Exception:
                        print("Không thể mở LONG")
            except Exception as e:
                print(f"Lỗi khối mở lệnh: {e}")

        # Đóng LONG khi action == 1 (giữ như code gốc)
        if vithedangmo and action == 1:
            try:
                donglong(SYMBOL, QTY)
                bot.sendMessage(TELEGRAM_CHAT_ID, "donglong")
            except Exception:
                print("Không thể đóng LONG")
            try:
                cancelallorder(SYMBOL)
            except Exception:
                print("Không thể xoá lệnh")

        # Nghỉ 5 phút
        time.sleep(300)
