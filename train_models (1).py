
import pandas as pd
import numpy as np

def engineer_features_for_training(df_raw_data, ticker_symbol):
    df_processed = df_raw_data.copy()
    ticker_upper = ticker_symbol.upper()

    # Normalize column names to 'Metric_TICKER' format
    # This handles both single-level and MultiIndex columns
    if isinstance(df_processed.columns, pd.MultiIndex):
        if df_processed.columns.nlevels == 3 and 'Ticker' in df_processed.columns.names:
            df_processed.columns = [f'{col[2]}_{col[1]}' for col in df_processed.columns.values]
        elif df_processed.columns.nlevels == 2:
            df_processed.columns = [f'{col[0]}_{col[1]}' for col in df_processed.columns.values]
        else:
            df_processed.columns = ['_'.join(map(str, col)).strip() for col in df_processed.columns.values]

    desired_metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
    rename_map = {}
    for col in df_processed.columns:
        if col in desired_metrics and not col.endswith(f'_{ticker_upper}'):
            rename_map[col] = f'{col}_{ticker_upper}'

    df_processed.rename(columns=rename_map, inplace=True)

    final_cols_prefixed = [f'{metric}_{ticker_upper}' for metric in desired_metrics]

    existing_final_cols = [col for col in final_cols_prefixed if col in df_processed.columns]
    if len(existing_final_cols) < len(final_cols_prefixed):
        print(f"Warning: Some expected columns for {ticker_upper} were not found after processing: {set(final_cols_prefixed) - set(existing_final_cols)}")

    df_processed = df_processed[existing_final_cols]

    df_processed['SMA_20'] = df_processed[f'Close_{ticker_upper}'].rolling(window=20).mean()
    df_processed['SMA_50'] = df_processed[f'Close_{ticker_upper}'].rolling(window=50).mean()

    df_processed['Daily_Return'] = df_processed[f'Close_{ticker_upper}'].pct_change() * 100

    df_processed['Lag_1'] = df_processed[f'Close_{ticker_upper}'].shift(1)
    df_processed['Lag_5'] = df_processed[f'Close_{ticker_upper}'].shift(5)
    df_processed['Lag_10'] = df_processed[f'Close_{ticker_upper}'].shift(10)

    df_processed['High_Low_Diff'] = df_processed[f'High_{ticker_upper}'] - df_processed[f'Low_{ticker_upper}']
    df_processed['Open_Close_Diff'] = df_processed[f'Open_{ticker_upper}'] - df_processed[f'Close_{ticker_upper}']

    df_processed.dropna(inplace=True)

    return df_processed



import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from datetime import date

stock_tickers_for_training = ['AAPL', 'GOOGL', 'KO', 'MSFT', 'NKE']
start_date_training = '2020-01-01'
end_date_training = date.today().strftime('%Y-%m-%d')

print("Starting model training for multiple stocks...")

for ticker_symbol in stock_tickers_for_training:
    print(f"Processing stock: {ticker_symbol}")

    historical_data_single = yf.download(ticker_symbol, start=start_date_training, end=end_date_training)

    if historical_data_single.empty:
        print(f"No historical data found for {ticker_symbol}. Skipping.")
        continue

    df_processed_stock = engineer_features_for_training(historical_data_single, ticker_symbol)

    if df_processed_stock.empty:
        print(f"Not enough data to engineer features for {ticker_symbol}. Skipping.")
        continue

    target_column = f'Close_{ticker_symbol.upper()}'
    X = df_processed_stock.drop(target_column, axis=1)
    y = df_processed_stock[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    print(f"Training model for {ticker_symbol}...")
    model.fit(X_train, y_train)

    model_filename = f'random_forest_model_{ticker_symbol.lower()}.joblib'
    joblib.dump(model, model_filename)

    print(f"Model for {ticker_symbol} trained and saved as '{model_filename}'.")

print("All specified stock models have been processed and saved.")
