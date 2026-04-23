import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import date
import os
import logging

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not os.path.exists('models'):
    os.makedirs('models')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates technical indicators based on past data."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # 1. Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. Momentum & Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['High'] - df['Low']
    
    # 3. Relative Strength Index (RSI - 14 Day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. Lagged Features
    df['Lag_1_Return'] = df['Daily_Return'].shift(1)
    df['Lag_5_Return'] = df['Daily_Return'].shift(5)
    
    df.dropna(inplace=True)
    return df

stock_tickers = ['AAPL', 'GOOGL', 'KO', 'MSFT', 'NKE']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

logging.info("Starting model training (Target: Next Day's % Return)...")

for ticker in stock_tickers:
    logging.info(f"Processing {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if len(data) < 150:
        logging.warning(f"Not enough data for {ticker}. Skipping.")
        continue

    # Engineer Features
    df_features = engineer_features(data)
    
    # Target: Predict the NEXT DAY'S PERCENTAGE RETURN
    df_features['Target_Next_Return'] = df_features['Close'].pct_change().shift(-1)
    df_model = df_features.dropna()
    
    # Define Predictors
    predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 
                  'Daily_Return', 'Volatility', 'RSI_14', 'MACD', 'Signal_Line', 
                  'Lag_1_Return', 'Lag_5_Return']
    
    X = df_model[predictors]
    y = df_model['Target_Next_Return']
    
    # Professional Time-Series Split (Never shuffle time-series data)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train Gradient Boosting Model
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"  > Mean Absolute Error (Return %): {mae:.5f}")
    
    joblib.dump(model, f'models/model_{ticker.lower()}.joblib')

logging.info("All models trained and saved in 'models/' folder.")
