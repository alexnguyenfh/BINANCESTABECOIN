import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import pandas_ta as ta
import time
from binance.um_futures import UMFutures
from binance.client import Client
import os
import telepot
# ============ Cấu hình đơn giản ============

token = '6879315183:AAHOs0XQ5HtriBTf82hxNn6KrGWL6g3zJ00' # telegram token
receiver_id = 685322834 # https://api.telegram.org/bot<TOKEN>/getUpdates
bot = telepot.Bot(token)


FEE_RATE = 0.0005     # 0.05% mở + 0.05% đóng (áp dụng trong ENV RL)
ATR_LEN = 14
ATR_SL_MULT = 1.5     # SL = 1.5 * ATR (không thấp hơn min_sl_%)
RR_MULT = 2.0         # TP = RR_MULT * SL => R:R ≈ 1:2
MIN_SL_PCT = 0.003    # SL tối thiểu 0.3% giá vào (tránh quá sát)
TICK_ROUND_DEC = 1    # BNBUSDT tick ~ 0.1 -> làm tròn 1 chữ số



# ============ Chạy giữ nguyên cấu trúc code gốc ============
while True:
    # Lấy API từ env (khuyên dùng). Bạn có thể giữ hard-code như cũ nếu muốn.
    api_key = os.getenv("BINANCE_KEY", "5Osz1aHEnae9W18UZy1dgLXaCNhofJ6pM9zNheacXXBE1NyX6SFfJ05gZ2ry0xI1")
    secret_key = os.getenv("BINANCE_SECRET", "gPiSoULgHIf0hTANHEkdY9hMrX8Ic2DiBHedF3afjRjaxTzMjmjFHyYhpFX0zt9G")
    client = Client(api_key=api_key, api_secret=secret_key, tld="com")

    def is_position_open(symbol):
        try:
            positions = client.futures_account()['positions']
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                    return True
            return False
        except Exception as e:
            print(f"Lỗi: {e}")
            return False
    
    
    
    def cancelallorder (symbol):
    
        # Get all open futures orders
        open_orders = client.futures_get_open_orders(symbol=symbol)
        
        # Loop through open orders and cancel stop-loss and take-profit orders
        for order in open_orders:
            # Check if the order is a stop-loss or take-profit order
            if order['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                # Cancel the order
                client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                print(f"Canceled order: {order['orderId']} ({order['type']})")
    
    
    
    def moshort(loaicoin,soluong):
        client.futures_create_order(symbol=loaicoin, side = "SELL", type = "MARKET",quantity = soluong, positionSide="SHORT" )
        mucgiahientai = client.futures_symbol_ticker(symbol=loaicoin)
        mucgiahientai = round(float(mucgiahientai['price']),0)
        mucgiastoplose = round((mucgiahientai + 8),0)
        mucgiatakeprofit = round((mucgiahientai - 8),0)
        client.futures_create_order(symbol=loaicoin, side = "BUY", type = "STOP_MARKET", quantity = soluong,stopPrice = mucgiastoplose , positionSide="SHORT"  )
        client.futures_create_order(symbol=loaicoin, side = "BUY", type = "TAKE_PROFIT_MARKET", quantity = soluong,stopPrice = mucgiatakeprofit , positionSide="SHORT" )
        
    def dongshort (loaicoin,soluong):
        client.futures_create_order(symbol=loaicoin, side = "BUY", type = "MARKET",quantity = soluong, positionSide="SHORT" )
    def molong(loaicoin,soluong):
        client.futures_create_order(symbol=loaicoin, side = "BUY", type = "MARKET",quantity = soluong, positionSide="LONG" )
        mucgiahientai = client.futures_symbol_ticker(symbol=loaicoin)
        mucgiahientai = round(float(mucgiahientai['price']),0)
        mucgiastoplose = round((mucgiahientai - 8),0)
        mucgiatakeprofit = round((mucgiahientai + 8),0)
        client.futures_create_order(symbol=loaicoin, side = "SELL", type = "STOP_MARKET", quantity = soluong,stopPrice = mucgiastoplose , positionSide="LONG"  )
        client.futures_create_order(symbol=loaicoin, side = "SELL", type = "TAKE_PROFIT_MARKET", quantity = soluong,stopPrice = mucgiatakeprofit , positionSide="LONG" )
    def donglong (loaicoin,soluong):
        client.futures_create_order(symbol=loaicoin, side = "SELL", type = "MARKET",quantity = soluong, positionSide="LONG" )


    # =========================
    # PHẦN RL: thêm phí giao dịch (0.05% mở + 0.05% đóng)
    # =========================
    def get_data(loaicoin):
        clientfutu = UMFutures()
        symbol = 'BNBUSDT'  # (giữ nguyên như code gốc)
        interval = '5m'
        limit = 1500
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
        df = df.dropna()
        return df

    class TradingEnv(gym.Env):
        def __init__(self, df):
            super(TradingEnv, self).__init__()
            self.df = df
            self.current_step = 0
            self.balance = 1000  # Vốn ban đầu
            self.position = 0    # 0: không giữ, 1: đang giữ (long 1 đơn vị)
            self.entry_price = 0
            self.done = False
            self.reward = 0
            self.history = []  # Lịch sử giao dịch

            self.n_features = 13
            self.action_space = gym.spaces.Discrete(3)  # 0: Mua, 1: Bán, 2: Giữ
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.balance = 1000
            self.position = 0
            self.entry_price = 0
            self.done = False
            self.reward = 0
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
                    self.history.append((self.current_step, current_price, 'Buy', f'fee_open={open_fee:.6f}'))

            elif action == 1:  # Bán (đóng vị thế nếu đang hold)
                if self.position == 1:
                    gross = current_price - self.entry_price
                    close_fee = current_price * FEE_RATE  # phí đóng 0.05%
                    net = gross - close_fee
                    reward = net
                    self.balance += reward
                    self.position = 0
                    self.history.append((
                        self.current_step, current_price, 'Sell',
                        f'gross={gross:.6f}', f'fee_close={close_fee:.6f}', f'net={net:.6f}'
                    ))
                    self.entry_price = 0

            # Cập nhật trạng thái
            self.current_step += 1
            if self.current_step >= len(self.df) - 1:
                self.done = True

            obs = self._next_observation()
            return obs, reward, self.done, False, {}

    # ========== Huấn luyện PPO như code gốc ==========
    df = get_data("BNBUSDT")
    assert all(col in df.columns for col in [
        'supertrend_direction', 'hieusoboll', 'rsi', 'bollinger_upper',
        'bollinger_lower', 'bollinger_mid', 'macd1', 'macd2',
        'xuhuongmacd', 'obv', 'adx'
    ]), "Missing required features in DataFrame df"

    env = TradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    # Kiểm tra mô hình với dữ liệu mới
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        if dones or truncated:
            break

    # Lưu mô hình đã huấn luyện
    model.save("ppo_BB"  + ".zip")

    # In ra lịch sử giao dịch
    print("Final Balance:", env.balance)
    print("Trading History:")
    for trade in env.history:
        print(trade)

    time.sleep(1)

    model.save("ppo_BB"  + ".zip")
    model = PPO.load("ppo_BB.zip")

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
    bot.sendMessage(receiver_id, "--------" + str(df["Close"].iloc[-1]))
    bot.sendMessage(receiver_id, "hanhdong " + str(df["Close"].iloc[-1]))

    print("--------")

    print("supertrend_direction:" + str(df['supertrend_direction'].iloc[-1]))
    print("xuhuongmacd :" +str(df['xuhuongmacd'].iloc[-1]))

    vithedangmo = is_position_open("BNBUSDT")
    print("vithedangmo :" +str(vithedangmo))
    if  df['Close'].iloc[-1] - df['Open'].iloc[-1]  > 0 :
        print("nen xanh")
    if  df['Close'].iloc[-1] - df['Open'].iloc[-1]  < 0 :
        print("nen do")
    print(df['Close'].iloc[-1])

    # Điều kiện vào lệnh của bạn (giữ nguyên), nhưng SL/TP đã chuyển sang ATR trong molong/moshort
    if env.balance > 1000 :

        if   vithedangmo  == False and action == 0 and  (df['xuhuongmacd'].iloc[-1]) > 0  and df['Close'].iloc[-1] - df['Open'].iloc[-1]  > 0:
           
            try:
                cancelallorder("BNBUSDT")
            except:
                print("ko the xoa lenh")
    
            try:
                molong("BNBUSDT",0.02)
                bot.sendMessage(receiver_id, "molong " + str(df["Close"].iloc[-1]))

            except:
                print("ko the mo long")
    if   vithedangmo  == True and action == 1  and  (df['xuhuongmacd'].iloc[-1]) < 0  and df['Close'].iloc[-1] - df['Open'].iloc[-1]  < 0: 
       
        try:
            donglong ("BNBUSDT",0.02)
            bot.sendMessage(receiver_id, "donglong " + str(df["Close"].iloc[-1]))

        except:
            print("ko the dong long")
        try:
            cancelallorder("BNBUSDT")
        except:
            print("ko the xoa lenh")

    time.sleep(300)
