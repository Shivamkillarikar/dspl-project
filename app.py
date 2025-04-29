import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import pytz

# API setup
API_KEY = '1afdd88fb14c4b25e2e9192d7eabbba9'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Functions
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'WindGustSpeed': data['wind']['speed']
    }

def read_historical_data(filename):
    df = pd.read_csv(filename)
    for col in ['Temp', 'Humidity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().drop_duplicates()

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append([data[feature].iloc[i]])
        y.append(data[feature].iloc[i + 1])
    return np.array(X), np.array(y)

def train_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # SVR
    svr_pipeline = make_pipeline(StandardScaler(), SVR())
    svr_params = {
        'svr__C': [0.1, 1, 10],
        'svr__gamma': ['scale', 'auto', 0.01],
        'svr__epsilon': [0.1, 0.2, 0.5],
        'svr__kernel': ['linear', 'rbf']
    }
    svr_search = RandomizedSearchCV(svr_pipeline, svr_params, n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)
    svr_search.fit(X_train, y_train)
    best_svr = svr_search.best_estimator_

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    models = {
        "RandomForest": best_rf,
        "LinearRegression": lr,
        "SVR": best_svr
    }

    trained_models = {}
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        trained_models[name] = model
        metrics[name] = {"MSE": mse, "R2": r2}

    return trained_models, X_test, y_test, metrics

def predict_future(models, current_value):
    predictions = {name: [current_value] for name in models}
    for _ in range(5):
        for name, model in models.items():
            next_value = model.predict(np.array([[predictions[name][-1]]]))[0]
            predictions[name].append(next_value)
    return {name: values[1:] for name, values in predictions.items()}

def plot_actual_vs_predicted(models, X_test, y_test, title):
    plt.figure(figsize=(10, 5))
    for name, model in models.items():
        y_pred = model.predict(X_test)
        plt.plot(y_pred, label=f'Predicted - {name}')
    plt.plot(y_test, label='Actual', color='black', linewidth=2)
    plt.title(f'Actual vs Predicted {title}')
    plt.xlabel('Sample Index')
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

# Streamlit App
def main():
    st.title("🌤️ Weather Prediction App")

    city = st.text_input("Enter a city name", "Karachi")
    file_uploaded = st.file_uploader("Upload historical weather CSV file", type=['csv'])

    if city and file_uploaded:
        try:
            current_weather = get_current_weather(city)
            historical_data = read_historical_data(file_uploaded)

            # Prepare data
            X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
            X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

            # Train models
            with st.spinner('Training models for temperature...'):
                temp_models, X_temp_test, y_temp_test, temp_metrics = train_regression_models(X_temp, y_temp)

            with st.spinner('Training models for humidity...'):
                hum_models, X_hum_test, y_hum_test, hum_metrics = train_regression_models(X_hum, y_hum)

            # Future prediction
            future_temp = predict_future(temp_models, current_weather['temp_min'])
            future_humidity = predict_future(hum_models, current_weather['humidity'])

            timezone = pytz.timezone('Asia/Karachi')
            now = datetime.now(timezone)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

            # Display current weather
            st.subheader(f"Current Weather in {city}, {current_weather['country']}")
            st.metric("Temperature", f"{current_weather['current_temp']} °C")
            st.metric("Feels Like", f"{current_weather['feels_like']} °C")
            st.metric("Humidity", f"{current_weather['humidity']}%")
            st.info(f"Weather Condition: {current_weather['description']}")

            # Display predictions
            st.subheader("🌡️ Future Temperature Predictions")
            for model_name, preds in future_temp.items():
                st.write(f"**{model_name}**")
                df_temp_pred = pd.DataFrame({
                    "Time": future_times,
                    "Predicted Temp (°C)": [round(p, 1) for p in preds]
                })
                st.table(df_temp_pred)

            st.subheader("💧 Future Humidity Predictions")
            for model_name, preds in future_humidity.items():
                st.write(f"**{model_name}**")
                df_hum_pred = pd.DataFrame({
                    "Time": future_times,
                    "Predicted Humidity (%)": [round(p, 1) for p in preds]
                })
                st.table(df_hum_pred)

            # Plot actual vs predicted
            st.subheader("📈 Temperature Prediction Performance")
            plot_actual_vs_predicted(temp_models, X_temp_test, y_temp_test, "Temperature (°C)")

            st.subheader("📈 Humidity Prediction Performance")
            plot_actual_vs_predicted(hum_models, X_hum_test, y_hum_test, "Humidity (%)")

            # Model metrics
            st.subheader("🧠 Model Performance Metrics")
            st.write("**Temperature Models**")
            st.json(temp_metrics)
            st.write("**Humidity Models**")
            st.json(hum_metrics)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
