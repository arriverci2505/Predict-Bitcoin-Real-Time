import streamlit as st
import ccxt
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import time
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

# --- 1. CẤU TRÚC MODEL ---
class EnsembleModel:
    def __init__(self):
        self.models = {'gbr': GradientBoostingRegressor(), 'rf': RandomForestRegressor(), 'ridge': Ridge()}
        self.weights = None
        self.scaler = RobustScaler()
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            predictions += self.weights[name] * model.predict(X_scaled)
        return predictions

# --- 2. HÀM TÍNH TOÁN FEATURE (ĐƯA RA NGOÀI VÒNG LẶP) ---
def engineer_features(df):
    df = df.copy()
    close_prev = df['Close'].shift(1)
    high_prev = df['High'].shift(1)
    low_prev = df['Low'].shift(1)
    open_prev = df['Open'].shift(1)
    volume_prev = df['Volume'].shift(1)
    # 1. PRICE-BASED INDICATORS
    
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = close_prev.rolling(period).mean()
        df[f'SMA_Dist_{period}'] = (close_prev - df[f'SMA_{period}']) / close_prev
    
    # Exponential Moving Averages
    for period in [9, 12, 21, 26, 50]:
        df[f'EMA_{period}'] = close_prev.ewm(span=period, adjust=False).mean()
        df[f'EMA_Dist_{period}'] = (close_prev - df[f'EMA_{period}']) / close_prev
    
    # Moving Average Crossovers
    df['MA_Cross_Fast'] = (df['SMA_10'] - df['SMA_20']) / close_prev
    df['MA_Cross_Medium'] = (df['SMA_20'] - df['SMA_50']) / close_prev
    df['MA_Cross_Slow'] = (df['SMA_50'] - df['SMA_200']) / close_prev
    
    # Price momentum
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'Return_{period}'] = close_prev.pct_change(period)
        df[f'Log_Return_{period}'] = np.log(close_prev / close_prev.shift(period))
    
    # 2. MOMENTUM INDICATORS
    
    # RSI (Relative Strength Index)
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).shift(1).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).shift(1).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        df[f'RSI_{period}_Norm'] = (df[f'RSI_{period}'] - 50) / 50
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = close_prev.ewm(span=12, adjust=False).mean()
    ema_26 = close_prev.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].shift(1).ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Hist_Norm'] = df['MACD_Hist'] / close_prev
    
    # Stochastic Oscillator
    for period in [14, 21]:
        low_min = low_prev.rolling(period).min()
        high_max = high_prev.rolling(period).max()
        df[f'Stoch_{period}'] = 100 * (close_prev - low_min) / (high_max - low_min + 1e-10)
        df[f'Stoch_{period}_D'] = df[f'Stoch_{period}'].rolling(3).mean()
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df[f'ROC_{period}'] = (close_prev - close_prev.shift(period)) / close_prev.shift(period)
    
    # Williams %R
    high_14 = high_prev.rolling(14).max()
    low_14 = low_prev.rolling(14).min()
    df['Williams_R'] = -100 * (high_14 - close_prev) / (high_14 - low_14 + 1e-10)
    
    # 3. VOLATILITY INDICATORS
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    for period in [7, 14, 21]:
        df[f'ATR_{period}'] = true_range.shift(1).rolling(period).mean()
        df[f'ATR_{period}_Pct'] = df[f'ATR_{period}'] / close_prev
    
    # Bollinger Bands
    for period in [20, 50]:
        sma = close_prev.rolling(period).mean()
        std = close_prev.rolling(period).std()
        df[f'BB_Upper_{period}'] = sma + (std * 2)
        df[f'BB_Lower_{period}'] = sma - (std * 2)
        df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / sma
        df[f'BB_Position_{period}'] = (close_prev - df[f'BB_Lower_{period}']) / \
                                      (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'] + 1e-10)
    
    # Historical Volatility
    for period in [10, 20, 30]:
        df[f'HV_{period}'] = close_prev.pct_change().rolling(period).std() * np.sqrt(period)
    
    # Keltner Channels
    ema_20 = close_prev.ewm(span=20, adjust=False).mean()
    df['Keltner_Upper'] = ema_20 + (df['ATR_14'] * 2)
    df['Keltner_Lower'] = ema_20 - (df['ATR_14'] * 2)
    df['Keltner_Position'] = (close_prev - df['Keltner_Lower']) / \
                             (df['Keltner_Upper'] - df['Keltner_Lower'] + 1e-10)
    
    # 4. VOLUME INDICATORS
    
    # Volume Moving Averages
    for period in [5, 10, 20]:
        df[f'Volume_MA_{period}'] = volume_prev.rolling(period).mean()
        df[f'Volume_Ratio_{period}'] = volume_prev / (df[f'Volume_MA_{period}'] + 1e-10)
    
    # Volume Trend
    df['Volume_Trend'] = volume_prev.rolling(5).mean() / (volume_prev.rolling(20).mean() + 1e-10)
    
    # On-Balance Volume (OBV)
    obv = (volume_prev * np.sign(df['Close'].diff())).fillna(0).cumsum()
    df['OBV'] = obv.shift(1)
    df['OBV_MA'] = df['OBV'].rolling(20).mean()
    df['OBV_Ratio'] = df['OBV'] / (df['OBV_MA'] + 1e-10)
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).shift(1).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).shift(1).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 1e-10)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # Volume Price Trend (VPT)
    df['VPT'] = (volume_prev * df['Close'].pct_change()).fillna(0).cumsum().shift(1)
    
    # 5. CANDLESTICK PATTERNS
    
    # Basic candle metrics
    body = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    df['Body_Size'] = (body / (total_range + 1e-10)).shift(1)
    df['Upper_Shadow'] = (upper_shadow / (total_range + 1e-10)).shift(1)
    df['Lower_Shadow'] = (lower_shadow / (total_range + 1e-10)).shift(1)
    df['Body_Direction'] = np.sign(df['Close'] - df['Open']).shift(1)
    
    # Candle range relative to ATR
    df['Range_ATR_Ratio'] = (total_range / (df['ATR_14'] + 1e-10)).shift(1)
    
    # 6. PATTERN RECOGNITION (Simple)
    
    # Doji (thân nến rất nhỏ)
    df['Is_Doji'] = (df['Body_Size'] < 0.1).astype(int)
    
    # Hammer / Hanging Man
    df['Is_Hammer'] = ((df['Lower_Shadow'] > 2 * df['Body_Size']) & 
                       (df['Upper_Shadow'] < df['Body_Size'])).astype(int)
    
    # Shooting Star / Inverted Hammer
    df['Is_Shooting_Star'] = ((df['Upper_Shadow'] > 2 * df['Body_Size']) & 
                              (df['Lower_Shadow'] < df['Body_Size'])).astype(int)
    
    # 7. LAG FEATURES (Temporal patterns)
    
    # Previous candles returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df[f'Return_1'].shift(lag)
        df[f'RSI_14_Lag_{lag}'] = df['RSI_14'].shift(lag)
        df[f'Volume_Ratio_20_Lag_{lag}'] = df['Volume_Ratio_20'].shift(lag)
    
    # Consecutive up/down candles
    up_candle = (df['Close'] > df['Open']).astype(int).shift(1)
    df['Consecutive_Up'] = up_candle.rolling(5).sum()
    df['Consecutive_Down'] = (1 - up_candle).rolling(5).sum()

    # 8. TIME-BASED FEATURES
    
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    
    # Cyclical encoding
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Market session (crude approximation)
    df['Is_Asian_Session'] = ((df['Hour'] >= 0) & (df['Hour'] < 8)).astype(int)
    df['Is_European_Session'] = ((df['Hour'] >= 8) & (df['Hour'] < 16)).astype(int)
    df['Is_US_Session'] = ((df['Hour'] >= 16) & (df['Hour'] < 24)).astype(int)

    # 9. REGIME DETECTION
    
    # Trend strength
    for period in [20, 50, 100]:
        returns = close_prev.pct_change(period)
        volatility = close_prev.pct_change().rolling(period).std()
        df[f'Trend_Strength_{period}'] = returns / (volatility + 1e-10)
    
    # Volatility regime
    df['Volatility_Regime'] = df['HV_20'].rolling(50).mean()
    df['Volatility_Percentile'] = df['HV_20'].rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )
    
    # 10. TARGETS (Multiple timeframes)
    
    # Future returns (chỉ để training, không dùng làm feature)
    df['Target_Return_1'] = df['Close'].pct_change(1).shift(-1)   # Next candle
    df['Target_Return_3'] = df['Close'].pct_change(3).shift(-3)   # 3 candles
    df['Target_Return_5'] = df['Close'].pct_change(5).shift(-5)   # 5 candles
    df['Target_Return_10'] = df['Close'].pct_change(10).shift(-10) # 10 candles
    
    # Direction (binary classification)
    df['Target_Direction_1'] = (df['Target_Return_1'] > 0).astype(int)
    
    # Risk metrics
    df['Target_Max_Favorable'] = df['High'].rolling(3).max().shift(-3) / df['Close'] - 1
    df['Target_Max_Adverse'] = df['Low'].rolling(3).min().shift(-3) / df['Close'] - 1
    
    return df.copy()

# --- 3. HÀM LẤY DỮ LIỆU ---
def get_data():
    try:
        exchange = ccxt.kraken()
        # Tăng limit lên 300 để đủ dữ liệu tính SMA 200
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=300)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms') + timedelta(hours=7)
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối sàn: {e}")
        return pd.DataFrame()

# --- 4. GIAO DIỆN ---
st.set_page_config(page_title="BTC AI Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, "BTC_USD_ensemble.pkl"))
    with open(os.path.join(current_dir, "BTC_USD_features.txt"), 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

model, feature_cols = load_ai_model()
placeholder = st.empty()

# --- 5. VÒNG LẶP CHÍNH ---
while True:
    df_raw = get_data()

    if not df_raw.empty:
            df_features = engineer_features(df_raw)
            X_live = df_features[feature_cols].dropna().tail(1)

                # Lấy số phút hiện tại
            now = datetime.now()
            current_minute = now.minute
            first_run = True
            # Kiểm tra xem có phải là phút bắt đầu nến mới không (0, 15, 30, 45)
            is_new_candle = current_minute % 15 == 0
            
            # Logic trong vòng lặp:
            if is_new_candle or first_run:
                    if not X_live.empty:
                        prediction = model.predict(X_live.values)[0]
                        price = df_raw['Close'].iloc[-1]
                        
                        # --- LOGIC TÍN HIỆU (THRESHOLD) ---
                        # Ngưỡng để tránh nhiễu
                        threshold = 0.00025
                        tp, sl = 0.0, 0.0
                        
                        if prediction > 0.0008:
                            sig, col, icon = "STRONG BUY", "#00ff88", "🔥"
                            tp, sl = price * 1.006, price * 0.997
                        elif prediction > threshold:
                            sig, col, icon = "BUY", "#2ecc71", "📈"
                            tp, sl = price * 1.004, price * 0.998
                        elif prediction < -0.0008:
                            sig, col, icon = "STRONG SELL", "#ff4b4b", "💀"
                            tp, sl = price * 0.994, price * 1.003
                        elif prediction < -threshold:
                            sig, col, icon = "SELL", "#e74c3c", "📉"
                            tp, sl = price * 0.996, price * 1.002
                        else:
                            sig, col, icon = "HOLD", "#f1c40f", "⚖️"
                        
                        first_run = True
    
            # --- PHẦN HIỂN THỊ CHIA ĐÔI MÀN HÌNH ---
            with placeholder.container():
                # Chia làm 2 cột với tỷ lệ 1:1.2 (bên phải chart rộng hơn chút cho dễ nhìn)
                col_left, col_right = st.columns([1, 1.2])
            
                # --- CỘT TRÁI: QUẢN LÝ LỆNH ---
                with col_left:
                    st.subheader("🤖 Bitcoin Alpha: Neural Predictor")
                    st.markdown(f"""
                        <div style="background-color:{col}22; border: 2px solid {col}; padding:20px; border-radius:15px; text-align:center;">
                            <h1 style="color:{col}; margin:0; font-size: 35px;">{icon} {sig}</h1>
                            <h2 style="color:white; margin:10px 0;">${price:,.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
            
                    st.write("---")
                    
                    # Phần TP/SL và Chi tiết
                    if "HOLD" not in sig:
                        c1, c2 = st.columns(2)
                        c1.metric("🎯 Chốt lời (TP)", f"${tp:,.1f}")
                        c2.metric("⚠️ Cắt lỗ (SL)", f"${sl:,.1f}")
                        
                        st.markdown(f"**Cường độ dự báo:** `{prediction:+.6%}`")
                    else:
                        st.warning("⚖️ Hệ thống đang ở trạng thái mất cân bằng - Chờ tín hiệu rõ ràng hơn.")

                    st.caption(f"⏱️ Cập nhật cuối: {(datetime.now() + timedelta(hours = 7)).strftime('%H:%M:%S')}")
                    
                # --- CỘT PHẢI: TRADINGVIEW CHART ---
                with col_right:
                    st.markdown("### 📈 Real-time Market Chart")
                    # Mã nhúng TradingView
                    tv_widget = f"""
                        <div style="height:500px;">
                            <div id="tv_chart_main" style="height:100%;"></div>
                            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                            <script type="text/javascript">
                            new TradingView.widget({{
                                "autosize": true,
                                "symbol": "KRAKEN:BTCUSDT",
                                "interval": "15",
                                "timezone": "Asia/Ho_Chi_Minh",
                                "theme": "dark",
                                "style": "1",
                                "locale": "vi_VN",
                                "enable_publishing": false,
                                "allow_symbol_change": true,
                                "container_id": "tv_chart_main"
                            }});
                            </script>
                        </div>
                    """
                    st.components.v1.html(tv_widget, height=520)
    time.sleep(60)








