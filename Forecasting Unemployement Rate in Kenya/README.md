# Unemployment Rate Analysis and Forecasting

This project analyzes and forecasts the unemployment rate using time series techniques in Python. The data is fetched from the FRED (Federal Reserve Economic Data) database.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Stationarity Check](#stationarity-check)
- [Checking for Seasonality, Trend and Noise](#checking-for-seasonality-trend-and-noise)
- [Preparation for Modeling](#preparation-for-modeling)
- [Modeling](#modeling)
  - [ARIMA Model](#arima-model)
- [Future Predictions](#future-predictions)
- [Libraries Used](#libraries-used)

## Introduction

This notebook performs a time series analysis on unemployment rate data obtained from FRED. The goal is to understand the patterns in the data, check for stationarity and seasonality, and build an ARIMA model to forecast future unemployment rates.

## Dataset

The dataset used is the unemployment rate data for a specific country (indicated by the FRED series ID). The data is fetched directly from the FRED database using the `pandas_datareader` library.

## Data Preprocessing

This section involves:

- Fetching the data from FRED for a specified date range.
- Converting the data into a pandas DataFrame.
- Renaming columns for clarity.
- Converting the 'Date' column to datetime objects and setting it as the index.
- Checking for missing values.
- Ensuring the data is sorted by date.

## Exploratory Data Analysis

A time series plot of the unemployment rate over time is generated to visualize the trend and any apparent patterns in the data.

## Stationarity Check

The Augmented Dickey-Fuller (ADF) test is performed to check for stationarity in the time series data. Stationarity is an important assumption for many time series models like ARIMA.

## Checking for Seasonality, Trend and Noise

The time series is decomposed into its trend, seasonal, and residual components to visualize and understand these elements within the data.

## Preparation for Modeling

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots are generated to help identify potential parameters (p, d, q) for the ARIMA model.

## Modeling

### ARIMA Model

An ARIMA (AutoRegressive Integrated Moving Average) model is built to forecast the unemployment rate.

- The data is split into training and testing sets (80% training, 20% testing).
- An ARIMA model is defined with specific orders (p, d, q) based on the analysis of ACF and PACF plots.
- The model is fitted to the training data.
- Predictions are made on the test data.
- The model's performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
- A plot comparing the actual and predicted values on the test set is shown.

## Future Predictions

The fitted ARIMA model is used to forecast future unemployment rates for a specified number of steps (e.g., the next 36 periods). A plot shows the historical data along with the future forecast.

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib.pyplot`
- `seaborn`
- `pandas_datareader.data`
- `datetime`
- `statsmodels.tsa.seasonal`
- `statsmodels.tsa.stattools`
- `statsmodels.graphics.tsaplots`
- `statsmodels.tsa.statespace.sarimax`
- `sklearn.metrics`
