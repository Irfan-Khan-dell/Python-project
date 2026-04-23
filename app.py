import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import date, timedelta
import os

# --- SHARED FEATURE ENGINEERING FUNCTION ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['High'] - df['Low']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Lag_1_Return'] = df['Daily_Return'].shift(1)
    df['Lag_5_Return'] = df['Daily_Return'].shift(5)
    return df

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Stock Oracle", layout="wide")
st.title('📈 AI Stock Price Predictor')
st.markdown("""
This tool uses a **Gradient Boosting Regressor** trained to predict the **Next Day's Return** based on momentum, volatility, MACD, and RSI indicators.
""")

st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Stock Ticker", ('AAPL', 'GOOGL', 'KO', 'MSFT', 'NKE'))
start_date = st.sidebar.date_input("Data Start Date", value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input("Data End Date", value=date.today())

if st.sidebar.button('Run Prediction'):
    model_path = f'models/model_{ticker.lower()}.joblib'
    
    if not os.path.exists(model_path):
        st.error(f"Model for {ticker} not found. Please run 'train_Model.py' first.")
    else:
        model = joblib.load(model_path)
        with st.spinner(f'Fetching data for {ticker}...'):
            fetch_start = start_date - timedelta(days=150) # Buffer for 50-day SMA & RSI
            df_raw = yf.download(ticker, start=fetch_start, end=end_date + timedelta(days=1), progress=False)

        if df_raw.empty:
            st.error("No data found.")
        else:
            df_processed = engineer_features(df_raw)
            df_view = df_processed.loc[str(start_date):]
            
            predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 
                          'Daily_Return', 'Volatility', 'RSI_14', 'MACD', 'Signal_Line', 
                          'Lag_1_Return', 'Lag_5_Return']
            
            X = df_view[predictors].dropna()
            
            if X.empty:
                st.warning("Not enough data to generate predictions. Try an earlier start date.")
            else:
                # 1. Predict the RETURN percentage
                predicted_returns = model.predict(X)
                
                # 2. Convert predicted returns back to actual Dollar Prices
                actual_prices = df_view.loc[X.index, 'Close'].values
                predicted_prices = actual_prices * (1 + predicted_returns)
                
                # 3. Get Tomorrow's Prediction
                last_row_features = X.iloc[[-1]]
                future_return = model.predict(last_row_features)[0]
                current_price = df_view['Close'].iloc[-1]
                future_prediction_price = current_price * (1 + future_return)
                last_date = X.index[-1]
                
                # --- METRICS DISPLAY ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Latest Close", f"${current_price:.2f}", f"{last_date.date()}")
                
                # Show price difference
                price_diff = future_prediction_price - current_price
                col2.metric("Predicted Next Close", f"${future_prediction_price:.2f}", f"${price_diff:.2f}")
                
                # --- PLOTTING ---
                st.subheader("Backtesting: Predicted vs Actual (Next Day)")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot Actual Close
                ax.plot(X.index, actual_prices, label='Actual Close', color='black', alpha=0.6)
                
                # Plot Predicted Close (Shifted forward to the day it predicted)
                pred_series = pd.Series(predicted_prices, index=X.index).shift(1)
                ax.plot(pred_series.index, pred_series, label='Predicted Close', color='#00ff00', linestyle='--')
                
                ax.set_title(f"{ticker} - Model Performance (Gradient Boosting)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
