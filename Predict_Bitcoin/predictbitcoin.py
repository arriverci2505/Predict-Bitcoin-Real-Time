import streamlit as st
import ccxt
import pandas as pd
import joblib
import os
import gc
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
        # Model cần có cấu trúc giống hệt lúc bạn Train
        self.models_price = {}
        self.models_tp = {}
        self.models_sl = {}
        self.scaler = RobustScaler()

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        
        # Hàm này sẽ được joblib ghi đè khi load file .pkl
        # Tôi để đây để tránh lỗi cấu trúc Class
        return np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))

# --- 2. HÀM TÍNH TOÁN FEATURE ---
def engineer_features(df):
    df = df.copy()

    col = {}
    
    close_prev = df['Close'].shift(1)
    high_prev = df['High'].shift(1)
    low_prev = df['Low'].shift(1)
    open_prev = df['Open'].shift(1)
    volume_prev = df['Volume'].shift(1)
    # 1. PRICE-BASED INDICATORS
    
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        col[f'SMA_{period}'] = close_prev.rolling(period).mean()
        df[f'SMA_Dist_{period}'] = (close_prev - col[f'SMA_{period}']) / close_prev
    
    # Exponential Moving Averages
    for period in [9, 12, 21, 26, 50]:
        col[f'EMA_{period}'] = close_prev.ewm(span=period, adjust=False).mean()
        col[f'EMA_Dist_{period}'] = (close_prev - col[f'EMA_{period}']) / close_prev
    
    # Moving Average Crossovers
    col['MA_Cross_Fast'] = (col['SMA_10'] - col['SMA_20']) / close_prev
    col['MA_Cross_Medium'] = (col['SMA_20'] - col['SMA_50']) / close_prev
    col['MA_Cross_Slow'] = (col['SMA_50'] - col['SMA_200']) / close_prev
    
    # Price momentum
    for period in [1, 2, 3, 5, 10, 20]:
        col[f'Return_{period}'] = close_prev.pct_change(period)
        col[f'Log_Return_{period}'] = np.log(close_prev / close_prev.shift(period))
    
    # 2. MOMENTUM INDICATORS
    
    # RSI (Relative Strength Index)
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).shift(1).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).shift(1).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        col[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        col[f'RSI_{period}_Norm'] = (col[f'RSI_{period}'] - 50) / 50
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = close_prev.ewm(span=12, adjust=False).mean()
    ema_26 = close_prev.ewm(span=26, adjust=False).mean()
    col['MACD'] = ema_12 - ema_26
    col['MACD_Signal'] = col['MACD'].shift(1).ewm(span=9, adjust=False).mean()
    col['MACD_Hist'] = col['MACD'] - col['MACD_Signal']
    col['MACD_Hist_Norm'] = col['MACD_Hist'] / close_prev
    
    # Stochastic Oscillator
    for period in [14, 21]:
        low_min = low_prev.rolling(period).min()
        high_max = high_prev.rolling(period).max()
        col[f'Stoch_{period}'] = 100 * (close_prev - low_min) / (high_max - low_min + 1e-10)
        col[f'Stoch_{period}_D'] = col[f'Stoch_{period}'].rolling(3).mean()
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        col[f'ROC_{period}'] = (close_prev - close_prev.shift(period)) / close_prev.shift(period)
    
    # Williams %R
    high_14 = high_prev.rolling(14).max()
    low_14 = low_prev.rolling(14).min()
    col['Williams_R'] = -100 * (high_14 - close_prev) / (high_14 - low_14 + 1e-10)
    
    # 3. VOLATILITY INDICATORS
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    for period in [7, 14, 21]:
        col[f'ATR_{period}'] = true_range.shift(1).rolling(period).mean()
        col[f'ATR_{period}_Pct'] = col[f'ATR_{period}'] / close_prev
    
    # Bollinger Bands
    for period in [20, 50]:
        sma = close_prev.rolling(period).mean()
        std = close_prev.rolling(period).std()
        col[f'BB_Upper_{period}'] = sma + (std * 2)
        col[f'BB_Lower_{period}'] = sma - (std * 2)
        col[f'BB_Width_{period}'] = (col[f'BB_Upper_{period}'] - col[f'BB_Lower_{period}']) / sma
        col[f'BB_Position_{period}'] = (close_prev - col[f'BB_Lower_{period}']) / \
                                      (col[f'BB_Upper_{period}'] - col[f'BB_Lower_{period}'] + 1e-10)
    
    # Historical Volatility
    for period in [10, 20, 30]:
        col[f'HV_{period}'] = close_prev.pct_change().rolling(period).std() * np.sqrt(period)
    
    # Keltner Channels
    ema_20 = close_prev.ewm(span=20, adjust=False).mean()
    col['Keltner_Upper'] = ema_20 + (col['ATR_14'] * 2)
    col['Keltner_Lower'] = ema_20 - (col['ATR_14'] * 2)
    col['Keltner_Position'] = (close_prev - col['Keltner_Lower']) / \
                             (col['Keltner_Upper'] - col['Keltner_Lower'] + 1e-10)
    
    # 4. VOLUME INDICATORS
    
    # Volume Moving Averages
    for period in [5, 10, 20]:
        col[f'Volume_MA_{period}'] = volume_prev.rolling(period).mean()
        col[f'Volume_Ratio_{period}'] = volume_prev / (col[f'Volume_MA_{period}'] + 1e-10)
    
    # Volume Trend
    col['Volume_Trend'] = volume_prev.rolling(5).mean() / (volume_prev.rolling(20).mean() + 1e-10)
    
    # On-Balance Volume (OBV)
    obv = (volume_prev * np.sign(df['Close'].diff())).fillna(0).cumsum()
    col['OBV'] = obv.shift(1)
    col['OBV_MA'] = col['OBV'].rolling(20).mean()
    col['OBV_Ratio'] = col['OBV'] / (col['OBV_MA'] + 1e-10)
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).shift(1).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).shift(1).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 1e-10)
    col['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # Volume Price Trend (VPT)
    col['VPT'] = (volume_prev * df['Close'].pct_change()).fillna(0).cumsum().shift(1)
    
    # 5. CANDLESTICK PATTERNS
    
    # Basic candle metrics
    body = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    col['Body_Size'] = (body / (total_range + 1e-10)).shift(1)
    col['Upper_Shadow'] = (upper_shadow / (total_range + 1e-10)).shift(1)
    col['Lower_Shadow'] = (lower_shadow / (total_range + 1e-10)).shift(1)
    col['Body_Direction'] = np.sign(df['Close'] - df['Open']).shift(1)
    
    # Candle range relative to ATR
    col['Range_ATR_Ratio'] = (total_range / (col['ATR_14'] + 1e-10)).shift(1)
    
    # 6. PATTERN RECOGNITION (Simple)
    
    # Doji (thân nến rất nhỏ)
    col['Is_Doji'] = (col['Body_Size'] < 0.1).astype(int)
    
    # Hammer / Hanging Man
    col['Is_Hammer'] = ((col['Lower_Shadow'] > 2 * col['Body_Size']) & 
                       (col['Upper_Shadow'] < col['Body_Size'])).astype(int)
    
    # Shooting Star / Inverted Hammer
    col['Is_Shooting_Star'] = ((col['Upper_Shadow'] > 2 * col['Body_Size']) & 
                              (col['Lower_Shadow'] < col['Body_Size'])).astype(int)
    
    # 7. LAG FEATURES (Temporal patterns)
    
    # Previous candles returns
    for lag in [1, 2, 3, 5, 10]:
        col[f'Return_Lag_{lag}'] = col[f'Return_1'].shift(lag)
        col[f'RSI_14_Lag_{lag}'] = col['RSI_14'].shift(lag)
        col[f'Volume_Ratio_20_Lag_{lag}'] = col['Volume_Ratio_20'].shift(lag)
    
    # Consecutive up/down candles
    up_candle = (df['Close'] > df['Open']).astype(int).shift(1)
    col['Consecutive_Up'] = up_candle.rolling(5).sum()
    col['Consecutive_Down'] = (1 - up_candle).rolling(5).sum()

    # 8. TIME-BASED FEATURES
    
    col['Hour'] = df.index.hour
    col['DayOfWeek'] = df.index.dayofweek
    col['DayOfMonth'] = df.index.day
    
    # Cyclical encoding
    col['Hour_Sin'] = np.sin(2 * np.pi * col['Hour'] / 24)
    col['Hour_Cos'] = np.cos(2 * np.pi * col['Hour'] / 24)
    col['Day_Sin'] = np.sin(2 * np.pi * col['DayOfWeek'] / 7)
    col['Day_Cos'] = np.cos(2 * np.pi * col['DayOfWeek'] / 7)
    
    # Market session (crude approximation)
    col['Is_Asian_Session'] = ((col['Hour'] >= 0) & (col['Hour'] < 8)).astype(int)
    col['Is_European_Session'] = ((col['Hour'] >= 8) & (col['Hour'] < 16)).astype(int)
    col['Is_US_Session'] = ((col['Hour'] >= 16) & (col['Hour'] < 24)).astype(int)

    # 9. REGIME DETECTION
    
    # Trend strength
    for period in [20, 50, 100]:
        returns = close_prev.pct_change(period)
        volatility = close_prev.pct_change().rolling(period).std()
        df[f'Trend_Strength_{period}'] = returns / (volatility + 1e-10)
    
    # Volatility regime
    col['Volatility_Regime'] = col['HV_20'].rolling(50).mean()
    col['Volatility_Percentile'] = col['HV_20'].rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )

    # 10. TARGETS
    col['Target_High_Pct'] = (df['High'].shift(-1) - df['Close']) / df['Close'] # Mục tiêu TP
    col['Target_Low_Pct'] = (df['Low'].shift(-1) - df['Close']) / df['Close']  # Mục tiêu SL

    extra_features = pd.DataFrame(col, index=df.index)

    # 3. Dùng pd.concat để gộp tất cả vào df gốc trong MỘT LẦN DUY NHẤT
    df = pd.concat([df, extra_features], axis=1)
    
    return df.dropna().copy()

# --- 3. HÀM LẤY DỮ LIỆU ---
def get_data():
    try:
        exchange = ccxt.kraken()
        # Tăng limit lên 300 để đủ dữ liệu tính SMA 200
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=500)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms') + timedelta(hours=7)
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối sàn: {e}")
        return pd.DataFrame()

# --- 4. GIAO DIỆN ---
st.set_page_config(page_title="BTC AI Terminal", layout="wide")

# CSS để giao diện đẹp và không bị giật
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    [data-testid="stStatusWidget"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    # Xác định thư mục chứa file predictbitcoin.py hiện tại
    base_path = os.path.dirname(__file__)
    
    # Kết hợp với tên file để tạo đường dẫn tuyệt đối
    model_path = os.path.join(base_path, "BTC_USD_ensemble.pkl")
    features_path = os.path.join(base_path, "BTC_USD_features.txt")
    
    # Kiểm tra tồn tại để báo lỗi rõ ràng trên Streamlit
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại: {model_path}")
        st.stop()
        
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

model, feature_cols = load_ai_model()
# --- KHỞI TẠO FRAMEWORK GIAO DIỆN TĨNH ---
# Chia cột ngoài vòng lặp để Chart không bị load lại
col_left, col_right = st.columns([1, 1.2])

with col_right:
    st.markdown("### 📈 Real-time Market Chart")
    tv_widget = """
        <div style="height:550px;">
            <div id="tv_chart_main" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({
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
            });
            </script>
        </div>
    """
    components.html(tv_widget, height=570)

# Tạo placeholder CHỈ cho cột bên trái (Dự đoán AI)
with col_left:
    signal_placeholder = st.empty()

# --- 5. VÒNG LẶP CHÍNH ---
last_minute = -1
while True:
    now = datetime.now() + timedelta(hours=7)
    current_minute = now.minute

    if current_minute != last_minute:
        df_raw = get_data() 
        missing = set(feature_cols) - set(df_features.columns)
        if missing:
            st.warning(f"⚠️ Cảnh báo: Thiếu dữ liệu cho các cột: {missing}")
        if not df_raw.empty:
            df_features = engineer_features(df_raw)
            X_live = df_features[feature_cols].dropna().tail(1)
            
            if not X_live.empty:
                # Model mới trả về 3 mảng: (Xu hướng, Khoảng cách TP, Khoảng cách SL)
                pred_price, pred_tp_dist, pred_sl_dist = model.predict(X_live.values)
                
                # Lấy giá trị đầu tiên vì predict trả về list
                p_move = pred_price[0]
                p_tp = pred_tp_dist[0]
                p_sl = pred_sl_dist[0]
                price = df_raw['Close'].iloc[-1]
                
                # Logic phân loại tín hiệu (Giữ nguyên của bạn)
                threshold = 0.0002
                if prediction > 0.0008:
                    sig, col, icon = "STRONG BUY", "#00ff88", "🔥"
                    tp = price * (1 + p_tp)
                    sl = price * (1 + p_sl)
                elif prediction > threshold:
                    sig, col, icon = "BUY", "#2ecc71", "📈"
                    tp = price * (1 + p_tp)
                    sl = price * (1 + p_sl)
                elif prediction < -0.0008:
                    sig, col, icon = "STRONG SELL", "#ff4b4b", "💀"
                    tp = price * (1 + p_tp)
                    sl = price * (1 + p_sl)
                elif prediction < -threshold:
                    sig, col, icon = "SELL", "#e74c3c", "📉"
                    tp = price * (1 + p_tp)
                    sl = price * (1 + p_sl)
                else:
                    sig, col, icon = "HOLD", "#f1c40f", "⚖️"
                    tp, sl = 0.0, 0.0

                # --- CHỈ CẬP NHẬT PHẦN TÍN HIỆU ---
                with signal_placeholder.container():
                    st.subheader("🤖 Bitcoin Alpha: Neural Predictor")
                    st.markdown(f"""
                        <div style="background-color:{col}22; border: 2px solid {col}; padding:20px; border-radius:15px; text-align:center;">
                            <h1 style="color:{col}; margin:0; font-size: 35px;">{icon} {sig}</h1>
                            <h2 style="color:white; margin:10px 0;">${price:,.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("---")
                    
                    if "HOLD" not in sig:
                        c1, c2 = st.columns(2)
                        c1.metric("🎯 Chốt lời (TP)", f"${tp:,.1f}")
                        c2.metric("⚠️ Cắt lỗ (SL)", f"${sl:,.1f}")
                        st.markdown(f"**Cường độ dự báo:** `{prediction:+.6%}`")
                    else:
                        st.warning("⚖️ Hệ thống đang chờ tín hiệu rõ ràng hơn.")

                    st.caption(f"⏱️ Cập nhật: {now.strftime('%H:%M:%S')}")

        last_minute = current_minute
        # Đợi một chút để tránh vòng lặp chạy quá nhanh
        time.sleep(2.5)
    
    # Nghỉ 0.5 giây để tiết kiệm CPU nhưng vẫn bắt kịp giây 00
    time.sleep(0.5)

































