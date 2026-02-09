import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import date, timedelta
import os

# --- SHARED FEATURE ENGINEERING FUNCTION ---
# (Must match training script exactly)
def engineer_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_5'] = df['Close'].shift(5)
    df['Volatility'] = df['High'] - df['Low']
    return df

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Stock Oracle", layout="wide")
st.title('📈 AI Stock Price Predictor')
st.markdown("""
This tool uses a **Random Forest** model trained to predict the **Next Day's Closing Price** based on historical momentum and volatility.
""")

# Sidebar Inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Stock Ticker", ('AAPL', 'GOOGL', 'KO', 'MSFT', 'NKE'))
start_date = st.sidebar.date_input("Data Start Date", value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input("Data End Date", value=date.today())

if st.sidebar.button('Run Prediction'):
    model_path = f'models/model_{ticker.lower()}.joblib'
    
    if not os.path.exists(model_path):
        st.error(f"Model for {ticker} not found. Please run 'train_models.py' first.")
    else:
        # 1. Load Model & Data
        model = joblib.load(model_path)
        with st.spinner(f'Fetching data for {ticker}...'):
            # Fetch extra days to ensure moving averages can be calculated for the start date
            fetch_start = start_date - timedelta(days=100)
            df_raw = yf.download(ticker, start=fetch_start, end=end_date + timedelta(days=1), progress=False)

        if df_raw.empty:
            st.error("No data found.")
        else:
            # 2. Prepare Data
            df_processed = engineer_features(df_raw)
            
            # Filter to user selected range (cutting off the extra buffer we fetched)
            df_view = df_processed.loc[str(start_date):]
            
            # Define predictors (Must match training exactly)
            predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'Daily_Return', 'Lag_1', 'Lag_5', 'Volatility']
            
            # Drop rows with NaNs (caused by lags/SMAs)
            X = df_view[predictors].dropna()
            
            if X.empty:
                st.warning("Not enough data to generate predictions. Try an earlier start date.")
            else:
                # 3. Predict
                # Remember: The model predicts 'Next Day Close'. 
                # So if we feed it data from Day T, the result is the prediction for T+1.
                predictions = model.predict(X)
                
                # 4. Create Comparison DataFrame
                # Shift predictions to align with the FUTURE date they belong to
                # We simply map the prediction made at index `i` to the actual `Close` at index `i+1` for comparison
                
                compare_df = X.copy()
                compare_df['Actual_Close'] = df_view['Close'] # Today's Close
                compare_df['Predicted_Next_Close'] = predictions # Prediction for TOMORROW
                
                # To visualize "Actual vs Predicted", we shift the Actual column BACK so row T compares Pred(T) vs Actual(T)
                # But a cleaner way for charts: Plot Predicted(T) vs Actual(T) where T is the date
                
                # Let's get the NEXT DAY prediction (The "Oracle" part)
                last_row_features = X.iloc[[-1]] # The most recent data point
                future_prediction = model.predict(last_row_features)[0]
                last_date = X.index[-1]
                
                # --- METRICS DISPLAY ---
                col1, col2, col3 = st.columns(3)
                current_price = df_view['Close'].iloc[-1]
                
                col1.metric("Latest Close", f"${current_price:.2f}", f"{last_date.date()}")
                col2.metric("Predicted Next Close", f"${future_prediction:.2f}", delta=f"{future_prediction - current_price:.2f}")
                
                # --- PLOTTING ---
                st.subheader("Backtesting: Predicted vs Actual (Next Day)")
                
                # Align for plotting: We predicted T+1 on day T. 
                # Let's shift predictions forward by 1 day index to overlay them on the day they were supposed to happen.
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot Actual
                ax.plot(df_view.index, df_view['Close'], label='Actual Close', color='black', alpha=0.6)
                
                # Plot Predictions (Shifted forward to match the day they predict)
                # We create a time series shifted by 1 freq
                pred_series = pd.Series(predictions, index=X.index)
                pred_series = pred_series.shift(1) 
                
                ax.plot(pred_series.index, pred_series, label='Predicted Close', color='#00ff00', linestyle='--')
                
                ax.set_title(f"{ticker} - Model Performance")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Show raw data
                with st.expander("See Raw Data"):
                    st.dataframe(compare_df[['Close', 'Predicted_Next_Close']].tail(10))
