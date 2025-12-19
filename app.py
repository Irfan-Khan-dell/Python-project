import streamlit as st
import pandas as pd
# yfinance may not be installed in the runtime; handle that gracefully
try:
    import yfinance as yf
except ImportError:
    yf = None
import joblib
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import os

# 1. Define the engineer_features function (copied from previous steps)
def engineer_features(df_raw_data, ticker_symbol):
    df_processed = df_raw_data.copy()
    ticker_upper = ticker_symbol.upper()

    # Normalize column names to 'Metric_TICKER' format
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
        st.warning(f"Warning: Some expected columns for {ticker_upper} were not found after processing: {set(final_cols_prefixed) - set(existing_final_cols)}")

    # Only select columns that actually exist in the dataframe
    if not existing_final_cols:
        return pd.DataFrame(), pd.Series(dtype=float)

    df_processed = df_processed[existing_final_cols]

    # Ensure required columns exist before calculating features
    required_for_features = [f'Close_{ticker_upper}', f'High_{ticker_upper}', f'Low_{ticker_upper}', f'Open_{ticker_upper}']
    if not all(col in df_processed.columns for col in required_for_features):
        st.warning("Not all required raw metric columns are available to compute derived features.")
        return pd.DataFrame(), pd.Series(dtype=float)

    df_processed['SMA_20'] = df_processed[f'Close_{ticker_upper}'].rolling(window=20).mean()
    df_processed['SMA_50'] = df_processed[f'Close_{ticker_upper}'].rolling(window=50).mean()
    df_processed['Daily_Return'] = df_processed[f'Close_{ticker_upper}'].pct_change() * 100
    df_processed['Lag_1'] = df_processed[f'Close_{ticker_upper}'].shift(1)
    df_processed['Lag_5'] = df_processed[f'Close_{ticker_upper}'].shift(5)
    df_processed['Lag_10'] = df_processed[f'Close_{ticker_upper}'].shift(10)
    df_processed['High_Low_Diff'] = df_processed[f'High_{ticker_upper}'] - df_processed[f'Low_{ticker_upper}']
    df_processed['Open_Close_Diff'] = df_processed[f'Open_{ticker_upper}'] - df_processed[f'Close_{ticker_upper}']

    df_processed.dropna(inplace=True)

    feature_cols = [
        f'Open_{ticker_upper}', f'High_{ticker_upper}', f'Low_{ticker_upper}', f'Volume_{ticker_upper}',
        'SMA_20', 'SMA_50', 'Daily_Return', 'Lag_1', 'Lag_5', 'Lag_10',
        'High_Low_Diff', 'Open_Close_Diff'
    ]

    # Filter feature_cols to only include those present in df_processed
    X_processed = df_processed[[col for col in feature_cols if col in df_processed.columns]]
    y_actual = df_processed[f'Close_{ticker_upper}'] if f'Close_{ticker_upper}' in df_processed.columns else pd.Series(dtype=float)

    return X_processed, y_actual

# Streamlit App Interface
st.title('Stock Price Predictor')

st.write('This application predicts stock prices using pre-trained RandomForestRegressor models.')
st.write('Models are available for AAPL, GOOGL, KO, MSFT, NKE.')

# User input for stock ticker
ticker_symbol = st.text_input('Enter Stock Ticker Symbol (e.g., AAPL):', 'AAPL').upper()

# Default dates for prediction
today = date.today()
default_start_date = pd.to_datetime('2023-01-01').date()

# User input for date range
start_date = st.date_input('Start Date', value=default_start_date)
end_date = st.date_input('End Date', value=today)

# Get Prediction button
if st.button('Get Prediction'):
    # Check for yfinance availability first
    if yf is None:
        st.error("The Python package 'yfinance' is not installed in this environment. Install it with:\n\npip install yfinance\n\nOr install everything needed with:\n\npip install -r requirements.txt")
    else:
        model_filename = f"random_forest_model_{ticker_symbol.lower()}.joblib"

        if not os.path.exists(model_filename):
            st.error(f"No pre-trained model found for {ticker_symbol}. Please ensure a model for this ticker has been trained and saved.")
        else:
            try:
                # Load the correct model based on the ticker symbol
                model = joblib.load(model_filename)

                if start_date >= end_date:
                    st.error('Error: End date must be after start date.')
                else:
                    # Fetch historical data
                    historical_data = yf.download(ticker_symbol, start=start_date, end=end_date)

                    if historical_data.empty:
                        st.warning(f'No historical data found for {ticker_symbol} in the specified date range. Please check the ticker symbol and date range.')
                    else:
                        # Engineer features
                        X_processed, y_actual = engineer_features(historical_data, ticker_symbol)

                        if X_processed.empty:
                            st.warning('Not enough data to engineer features for prediction. Please ensure the date range is sufficient (e.g., at least 50 days).')
                        else:
                            # Make predictions
                            y_pred = model.predict(X_processed)
                            y_pred_series = pd.Series(y_pred, index=y_actual.index, name='Predicted Prices')

                            st.subheader(f'Actual vs. Predicted Stock Prices for {ticker_symbol}')

                            # Display chart
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(y_actual.index, y_actual, label='Actual Prices', color='blue', linewidth=2)
                            ax.plot(y_pred_series.index, y_pred_series, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
                            ax.set_xlabel('Date')
                            ax.set_ylabel('Stock Price')
                            ax.set_title(f'{ticker_symbol} Stock Price: Actual vs. Predicted')
                            ax.legend()
                            ax.tick_params(axis='x', rotation=45)
                            st.pyplot(fig)

                            # Display key metrics
                            st.subheader('Key Metrics')
                            if not y_actual.empty:
                                st.write(f"Last Actual Closing Price: ${y_actual.iloc[-1]:.2f}")
                            if not y_pred_series.empty:
                                st.write(f"Last Predicted Closing Price: ${y_pred_series.iloc[-1]:.2f}")

            except Exception as e:
                st.error(f"An error occurred: {e}. Please try again.")    # Only select columns that actually exist in the dataframe
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

    feature_cols = [
        f'Open_{ticker_upper}', f'High_{ticker_upper}', f'Low_{ticker_upper}', f'Volume_{ticker_upper}',
        'SMA_20', 'SMA_50', 'Daily_Return', 'Lag_1', 'Lag_5', 'Lag_10',
        'High_Low_Diff', 'Open_Close_Diff'
    ]

    # Filter feature_cols to only include those present in df_processed
    X_processed = df_processed[[col for col in feature_cols if col in df_processed.columns]]
    y_actual = df_processed[f'Close_{ticker_upper}']

    return X_processed, y_actual

# Streamlit App Interface
st.title('Stock Price Predictor')

st.write('This application predicts stock prices using pre-trained RandomForestRegressor models.')
st.write('Models are available for AAPL, GOOGL, KO, MSFT, NKE.')

# User input for stock ticker
ticker_symbol = st.text_input('Enter Stock Ticker Symbol (e.g., AAPL):', 'AAPL').upper()

# Default dates for prediction
today = date.today()
default_start_date = pd.to_datetime('2023-01-01').date()

# User input for date range
start_date = st.date_input('Start Date', value=default_start_date)
end_date = st.date_input('End Date', value=today)

# Get Prediction button
if st.button('Get Prediction'):
    model_filename = f"random_forest_model_{ticker_symbol.lower()}.joblib"

    if not os.path.exists(model_filename):
        st.error(f"No pre-trained model found for {ticker_symbol}. Please ensure a model for this ticker has been trained and saved.")
    else:
        try:
            # Load the correct model based on the ticker symbol
            model = joblib.load(model_filename)

            if start_date >= end_date:
                st.error('Error: End date must be after start date.')
            else:
                # Fetch historical data
                historical_data = yf.download(ticker_symbol, start=start_date, end=end_date)

                if historical_data.empty:
                    st.warning(f'No historical data found for {ticker_symbol} in the specified date range. Please check the ticker symbol and date range.')
                else:
                    # Engineer features
                    X_processed, y_actual = engineer_features(historical_data, ticker_symbol)

                    if X_processed.empty:
                        st.warning('Not enough data to engineer features for prediction. Please ensure the date range is sufficient (e.g., at least 50 days).')
                    else:
                        # Make predictions
                        y_pred = model.predict(X_processed)
                        y_pred_series = pd.Series(y_pred, index=y_actual.index, name='Predicted Prices')

                        st.subheader(f'Actual vs. Predicted Stock Prices for {ticker_symbol}')

                        # Display chart
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(y_actual.index, y_actual, label='Actual Prices', color='blue', linewidth=2)
                        ax.plot(y_pred_series.index, y_pred_series, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Stock Price')
                        ax.set_title(f'{ticker_symbol} Stock Price: Actual vs. Predicted')
                        ax.legend()
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)

                        # Display key metrics
                        st.subheader('Key Metrics')
                        if not y_actual.empty:
                            st.write(f"Last Actual Closing Price: ${y_actual.iloc[-1]:.2f}")
                        if not y_pred_series.empty:
                            st.write(f"Last Predicted Closing Price: ${y_pred_series.iloc[-1]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}. Please try again.")
