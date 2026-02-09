import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from datetime import date
import os

# Create directory for models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

def engineer_features(df, ticker):
    """
    Creates technical indicators based on PAST data only.
    """
    df = df.copy()
    
    # Ensure simple column names (removes MultiIndex if present)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Calculate Technical Indicators
    # 1. Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. Daily Return (Momentum)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 3. Lagged Features (What happened yesterday?)
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_5'] = df['Close'].shift(5)
    
    # 4. Volatility (High - Low)
    df['Volatility'] = df['High'] - df['Low']
    
    # Drop NaN values created by rolling/shifting
    df.dropna(inplace=True)
    
    return df

stock_tickers = ['AAPL', 'GOOGL', 'KO', 'MSFT', 'NKE']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

print("Starting model training (Target: Next Day's Close)...")

for ticker in stock_tickers:
    print(f"\nProcessing {ticker}...")
    
    # Download Data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if len(data) < 100:
        print(f"Not enough data for {ticker}. Skipping.")
        continue

    # 1. Engineer Features (X)
    df_features = engineer_features(data, ticker)
    
    # 2. Create Target (y) - SHIFT BACKWARDS by 1
    # We want features at row 't' to predict Close at row 't+1'
    df_features['Target_Next_Close'] = df_features['Close'].shift(-1)
    
    # Drop the very last row because it has no 'Target_Next_Close' (we don't know tomorrow yet)
    df_model = df_features.dropna()
    
    # Define Predictors (X) and Target (y)
    predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'Daily_Return', 'Lag_1', 'Lag_5', 'Volatility']
    X = df_model[predictors]
    y = df_model['Target_Next_Close']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    print(f"  > R^2 Score: {score:.4f}")
    
    # Save
    joblib.dump(model, f'models/model_{ticker.lower()}.joblib')

print("\nAll models trained and saved in 'models/' folder.")
