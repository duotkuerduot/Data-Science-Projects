import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import datetime
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")
import argparse

def fetch_unemployment_data(start_date, end_date, fred_series_id="KENNGDPRPCPCPPPT"):
    """Fetches unemployment rate data from FRED."""
    print(f"Fetching data from FRED for series: {fred_series_id}")
    unemployment = web.DataReader(fred_series_id, "fred", start_date, end_date)
    unemployment = pd.DataFrame(unemployment)
    unemployment.reset_index(inplace=True)
    unemployment.columns = ['Date', 'Unemployment Rate']
    unemployment['Date'] = pd.to_datetime(unemployment['Date'])
    unemployment.set_index('Date', inplace=True)
    print("Data fetched successfully.")
    return unemployment

def preprocess_data(df):
    """Performs basic data preprocessing."""
    print("Preprocessing data...")
    # In a real deployment, you might handle missing values more robustly
    # For this example, we just check
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Warning: Found missing values:\n{missing_values}")

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    print("Data preprocessing complete.")
    return df

def plot_time_series(df, output_path="unemployment_rate_plot.png"):
    """Plots the unemployment rate time series and saves it."""
    print("Generating time series plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Unemployment Rate'], color='blue')
    plt.title('Unemployment Rate Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Unemployment Rate', fontsize=14)
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"Time series plot saved to {output_path}")

def adf_test(series):
    """Performs the Augmented Dickey-Fuller test."""
    print("Performing ADF test...")
    result = adfuller(series, autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', '#Observations Used']
    for value, label in zip(result, labels):
        print(f'{label} : {value}')
    if result[1] <= 0.05:
        print("Reject the null hypothesis: The time series is stationary")
    else:
        print("Fail to reject the null hypothesis: The time series is non-stationary")

def plot_decomposition(df, output_prefix="decomposition_plot"):
    """Decomposes the time series and plots the components."""
    print("Performing time series decomposition...")
    # Handle potential errors with decomposition if data is too short or has NaNs
    try:
        decomposition = seasonal_decompose(df['Unemployment Rate'], model='additive')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4)
        plt.subplot(411)
        plt.plot(df['Unemployment Rate'], label='Original')
        plt.legend(loc='upper left')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='upper left')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonality')
        plt.legend(loc='upper left')
        plt.subplot(414)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.close()
        print(f"Decomposition plot saved to {output_prefix}.png")
    except Exception as e:
        print(f"Error during decomposition plot: {e}")


def plot_acf_pacf(series, lags=12, output_prefix="acf_pacf_plots"):
    """Plots ACF and PACF of the time series."""
    print("Generating ACF and PACF plots...")
    try:
        plt.figure(figsize=(12, 6))
        plot_acf(series.dropna(), lags=lags) # Drop NaNs for plotting
        plt.title('ACF of Unemployment Rate', fontsize= 16)
        plt.xlabel('lags', fontsize=14)
        plt.ylabel('ACF', fontsize=14)
        plt.savefig(f"{output_prefix}_acf.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plot_pacf(series.dropna(), lags=lags) # Drop NaNs for plotting
        plt.title('PACF Of Unemployment Rate', fontsize= 16)
        plt.xlabel('lags', fontsize=14)
        plt.ylabel('PACF', fontsize=14)
        plt.savefig(f"{output_prefix}_pacf.png")
        plt.close()
        print(f"ACF and PACF plots saved to {output_prefix}_acf.png and {output_prefix}_pacf.png")
    except Exception as e:
        print(f"Error during ACF/PACF plotting: {e}")


def train_and_evaluate_arima(df, train_split=0.8, order=(1, 1, 1), output_path="arima_forecast_vs_actual.png"):
    """Trains, evaluates, and plots ARIMA model predictions."""
    print("Training and evaluating ARIMA model...")
    train_size = int(len(df) * train_split)
    train_data, test_data = df['Unemployment Rate'][0:train_size], df['Unemployment Rate'][train_size:]

    if len(train_data) == 0 or len(test_data) == 0:
        print("Error: Insufficient data for training or testing.")
        return None

    try:
        model = SARIMAX(train_data, order=order)
        arima_model = model.fit()

        predictions = arima_model.predict(start=len(train_data), end=len(df)-1)

        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        mape = mean_absolute_percentage_error(test_data, predictions)

        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Absolute Percentage Error (MAPE): {mape}')

        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label='Training Data')
        plt.plot(test_data.index, test_data, label='Actual Test Data')
        plt.plot(predictions.index, predictions, label='ARIMA Predictions', color='red')
        plt.title('ARIMA Model Forecast vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Unemployment Rate')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"ARIMA forecast vs actual plot saved to {output_path}")

        return arima_model
    except Exception as e:
        print(f"Error during ARIMA training or evaluation: {e}")
        return None

def forecast_future(model, df, steps=36, output_path="arima_future_forecast.png"):
    """Forecasts future values using the trained ARIMA model."""
    if model is None:
        print("Cannot forecast future, ARIMA model training failed.")
        return

    print(f"Forecasting future values for {steps} steps...")
    try:
        future_forecast = model.forecast(steps=steps)

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Unemployment Rate'], label='Historical Data')
        plt.plot(future_forecast.index, future_forecast, label='Future Forecast', color='green')
        plt.title('ARIMA Model Future Forecast')
        plt.xlabel('Date')
        plt.ylabel('Unemployment Rate')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"Future forecast plot saved to {output_path}")
    except Exception as e:
        print(f"Error during future forecasting: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unemployment Rate Analysis and ARIMA Forecasting")
    parser.add_argument("--start_date", type=str, default="2000-01-01", help="Start date for data fetching (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-12-31", help="End date for data fetching (YYYY-MM-DD)")
    parser.add_argument("--fred_series_id", type=str, default="KENNGDPRPCPCPPPT", help="FRED series ID for unemployment rate")
    parser.add_argument("--arima_order_p", type=int, default=1, help="ARIMA order p")
    parser.add_argument("--arima_order_d", type=int, default=1, help="ARIMA order d")
    parser.add_argument("--arima_order_q", type=int, default=1, help="ARIMA order q")
    parser.add_argument("--forecast_steps", type=int, default=36, help="Number of steps to forecast into the future")

    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    arima_order = (args.arima_order_p, args.arima_order_d, args.arima_order_q)

    # Fetch and preprocess data
    unemployment_data = fetch_unemployment_data(start, end, args.fred_series_id)
    if unemployment_data.empty:
        print("Error: No data fetched from FRED. Please check the date range or series ID.")
    else:
        unemployment_data = preprocess_data(unemployment_data)

        # Exploratory Data Analysis and Checks
        plot_time_series(unemployment_data)
        adf_test(unemployment_data['Unemployment Rate'].dropna()) # Drop NaNs before ADF test
        plot_decomposition(unemployment_data)
        plot_acf_pacf(unemployment_data['Unemployment Rate'])

        # Modeling
        arima_model = train_and_evaluate_arima(unemployment_data, order=arima_order)

        # Future Predictions
        if arima_model: # Only forecast if the model was trained successfully
            forecast_future(arima_model, unemployment_data, steps=args.forecast_steps)

        print("Script finished.")