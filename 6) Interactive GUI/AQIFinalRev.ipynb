import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess data
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    return data

# Streamlit dashboard
st.title('Air Quality Forecasting Dashboard')

# Sidebar for user input
st.sidebar.title("Upload and Select Model")
file_path = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

if file_path is not None:
    data = load_data(file_path)
    data = data.rename(columns={'date': 'ds', 'PM2.5_24hr_avg': 'y'})
    
    # Group data by location and monitoring station
    grouped = data.groupby(['location', 'location_monitoring_station'])

    # Select location and monitoring station
    location = st.selectbox('Select Location', data['location'].unique())
    station = st.selectbox('Select Monitoring Station', data.loc[data['location'] == location, 'location_monitoring_station'].unique())

    # Filter data for selected location and station
    filtered_data = grouped.get_group((location, station)).reset_index(drop=True)

    # Prophet Model
    st.subheader('Prophet Model')
    model = Prophet()
    model.fit(filtered_data)

    # Make future predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Plot historical data and forecast
    st.subheader('Historical Data and Forecast (Prophet)')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    # Plot forecast components
    st.subheader('Forecast Components (Prophet)')
    fig2 = plot_components_plotly(model, forecast)
    st.plotly_chart(fig2)

    # Make predictions on future dates
    st.subheader('Make Predictions (Prophet)')
    future_dates = st.date_input('Select future dates', value=[])
    if future_dates:
        future_data = pd.DataFrame({'ds': future_dates})
        future_forecast = model.predict(future_data)
        st.write(future_forecast[['ds', 'yhat']])

    # Retrain the model
    if st.button('Retrain Prophet Model'):
        model = Prophet()
        model.fit(filtered_data)
        st.write('Prophet model retrained successfully!')

    # ARIMA Model
    st.sidebar.title("ARIMA Forecasting")
    ts_data = filtered_data[['ds', 'y']]
    ts_data.set_index('ds', inplace=True)

    # Train-test split
    train_size = st.sidebar.slider("Training data size", 0.1, 0.9, 0.8, 0.1)
    train_data = ts_data[:int(train_size * len(ts_data))]
    test_data = ts_data[int(train_size * len(ts_data)):]

    # Fit ARIMA model
    p = st.sidebar.slider("p (AR order)", 0, 5, 5, 1)
    d = st.sidebar.slider("d (Differencing order)", 0, 2, 1, 1)
    q = st.sidebar.slider("q (MA order)", 0, 5, 0, 1)
    order = (p, d, q)
    model_arima = ARIMA(train_data, order=order)
    model_fit = model_arima.fit()

    # Forecast
    forecast_arima = model_fit.forecast(steps=len(test_data))

    # Visualize the results
    st.subheader('Historical Data and Forecast (ARIMA)')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_data, label='Training Data')
    ax.plot(test_data, label='Testing Data')
    ax.plot(test_data.index, forecast_arima, label='Forecast', color='black')
    ax.legend()
    ax.set_title('ARIMA Forecasting')
    st.pyplot(fig)

    # Drop rows with missing values
    test_data = test_data.dropna()
    forecast_arima = forecast_arima.dropna()

    # Align the lengths of test_data and forecast
    min_length = min(len(test_data), len(forecast_arima))
    test_data = test_data[:min_length]
    forecast_arima = forecast_arima[:min_length]

    
    # Make predictions on future dates for ARIMA
    st.subheader('Make Predictions (ARIMA)')
    future_dates_arima = st.date_input('Select future dates for ARIMA', value=[], key='arima_dates')
    if future_dates_arima:
        future_data_arima = pd.DataFrame({'ds': future_dates_arima})
        future_data_arima.set_index('ds', inplace=True)
        future_forecast_arima = model_fit.forecast(steps=len(future_data_arima))
        st.write(future_forecast_arima)

    # Retrain the ARIMA model
    if st.button('Retrain ARIMA Model'):
        model_arima = ARIMA(train_data, order=order)
        model_fit = model_arima.fit()
        st.write('ARIMA model retrained successfully!')

    # Comparison of Prophet and ARIMA
    st.subheader('Comparison of Prophet and ARIMA Models')
    mae_prophet = mean_absolute_error(test_data, forecast['yhat'][:len(test_data)])
    st.write(f"Prophet Mean Absolute Error: {mae_prophet:.2f}")

    # Calculate R-squared score for Prophet
    r2_prophet = r2_score(test_data, forecast['yhat'][:len(test_data)])
    st.write(f"Prophet R-squared Score: {r2_prophet:.2f}")

    # Calculate Mean Absolute Error for ARIMA
    mae_arima = mean_absolute_error(test_data, forecast_arima)
    st.write(f"ARIMA Mean Absolute Error: {mae_arima:.2f}")

    # Calculate R-squared score for ARIMA
    r2_arima = r2_score(test_data, forecast_arima)
    st.write(f"ARIMA R-squared Score: {r2_arima:.2f}")


    st.write("---")
    st.write("**Insights and Understanding:**")
    st.write("- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction. Lower MAE indicates better model performance.")
    st.write("- **R-squared Score (R²)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. Higher R² indicates better model performance.")
    st.write("- **Prophet vs. ARIMA**: Compare the MAE and R² scores of both models to determine which model performs better for your data. A lower MAE and higher R² score indicate a more accurate model.")

# Output explanations
    st.write("---")
    st.write("**Explanations:**")
    st.write("- **Training data size**: Determines the proportion of the time series data that will be used for training the ARIMA model. A value between 0 and 1, where 0.8 (or 80%) is a common choice for splitting the data into training and testing sets.")
    st.write("- **p (AR order)**: Represents the order of the Autoregressive (AR) component of the ARIMA model. The AR component captures the influence of past values on the current value of the time series. A higher value of `p` means that more past values are considered in the model, which can capture more complex patterns but may also lead to overfitting.")
    st.write("- **d (Differencing order)**: Represents the order of differencing applied to the time series to make it stationary (i.e., remove trends and seasonality). Differencing is the process of subtracting the previous value from the current value to remove trends or patterns that violate the stationarity assumption of ARIMA models. A value of 0 means no differencing is applied, while a value of 1 or 2 is common for non-stationary time series data.")
    st.write("- **q (MA order)**: Represents the order of the Moving Average (MA) component of the ARIMA model. The MA component captures the influence of past errors or residuals on the current value of the time series. A higher value of `q` means that more past errors are considered in the model, which can capture more complex patterns but may also lead to overfitting.")

else:
    st.warning("Please upload an Excel file to get started.")