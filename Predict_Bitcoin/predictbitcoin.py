import streamlit as st
import ccxt
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime, timedelta
import time

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="BTC AI Signal", page_icon="📈")

# --- HÀM TÍNH TOÁN CHỈ BÁO (Copy hàm engineer_features của bạn vào đây) ---
def engineer_features(df):
    # ... (Giữ nguyên toàn bộ nội dung hàm engineer_features bạn đã viết) ...
    return df

# --- HÀM LẤY DỮ LIỆU ---
def get_data():
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=500)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms') + timedelta(hours=7)
        df.set_index('Date', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except:
        return pd.DataFrame()

# --- GIAO DIỆN STREAMLIT ---
st.title("🤖 BTC/USDT AI Trading Signal")
st.write("Khung thời gian: **15 Phút** | Sàn: **Kraken**")

# Load Model
@st.cache_resource
def load_ai_model():
    # Lấy đường dẫn hiện tại của file code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(current_dir, "BTC_USD_ensemble.pkl")
    features_path = os.path.join(current_dir, "BTC_USD_features.txt")
    
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

model, feature_cols = load_ai_model()

# Vùng cập nhật dữ liệu
placeholder = st.empty()

while True:
    df_raw = get_data()
    if not df_raw.empty:
        df_features = engineer_features(df_raw.copy())
        X_live = df_features[feature_cols]
        latest_row = X_live.dropna().tail(1)

        if not latest_row.empty:
            prediction = model.predict(latest_row.values)[0]
            current_price = df_raw['Close'].iloc[-1]
            
            # Tính TP/SL (Chốt lời 0.3%, Cắt lỗ 0.2%)
            if prediction > 0:
                signal, color, icon = "MUA (LONG)", "#2ecc71", "🚀"
                tp, sl = current_price * 1.003, current_price * 0.998
            else:
                signal, color, icon = "BÁN (SHORT)", "#e74c3c", "🔻"
                tp, sl = current_price * 0.997, current_price * 1.002

            with placeholder.container():
                # Hiển thị giá và tín hiệu
                st.markdown(f"""
                <div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center; color:white;">
                    <h1 style="margin:0;">{icon} {signal}</h1>
                    <h2 style="margin:0;">${current_price:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)

                # Hiển thị TP/SL
                col1, col2 = st.columns(2)
                col1.metric("🎯 Chốt lời (TP)", f"${tp:,.2f}")
                col2.metric("⚠️ Cắt lỗ (SL)", f"${sl:,.2f}")
                
                st.write(f"⏱️ Cập nhật lúc: {datetime.now().strftime('%H:%M:%S')}")
                st.write(f"📊 Cường độ dự báo: `{prediction:+.4%}`")


    time.sleep(60) # Cập nhật mỗi phút một lần để tiết kiệm tài nguyên
