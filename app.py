import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pytz

# --- Constants ---
API_KEY = '1afdd88fb14c4b25e2e9192d7eabbba9'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# --- Functions ---
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

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append([data[feature].iloc[i]])
        y.append(data[feature].iloc[i + 1])
    return np.array(X), np.array(y)

def train_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

def predict_future(models, current_value):
    predictions = {name: [current_value] for name in models}
    for _ in range(5):
        for name, model in models.items():
            next_value = model.predict(np.array([[predictions[name][-1]]]))[0]
            predictions[name].append(next_value)
    return {name: values[1:] for name, values in predictions.items()}

# --- Streamlit UI ---
st.set_page_config("Weather Predictor", layout="centered")
st.title("🌦️ Weather Predictor App")

city = st.text_input("Enter a city name:", value="Karachi")
file_path = st.text_input("Enter full path to historical weather CSV file:", value="weather.csv")

if city and file_path:
    try:
        with st.spinner("Processing..."):
            weather = get_current_weather(city)
            df = pd.read_csv(file_path).dropna().drop_duplicates()

            X_temp, y_temp = prepare_regression_data(df, 'Temp')
            X_hum, y_hum = prepare_regression_data(df, 'Humidity')

            temp_models = train_regression_models(X_temp, y_temp)
            hum_models = train_regression_models(X_hum, y_hum)

            timezone = pytz.timezone('Asia/Karachi')
            now = datetime.now(timezone)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            time_labels = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

            future_temp = predict_future(temp_models, weather['temp_min'])
            future_hum = predict_future(hum_models, weather['humidity'])

        st.subheader(f"📍 Current Weather in {weather['city']}, {weather['country']}")
        st.write(f"**Temperature:** {weather['current_temp']}°C (Feels like {weather['feels_like']}°C)")
        st.write(f"**Humidity:** {weather['humidity']}%")
        st.write(f"**Condition:** {weather['description'].capitalize()}")
        st.write(f"**Wind Speed:** {weather['WindGustSpeed']} m/s | Direction: {weather['wind_gust_dir']}°")
        st.write(f"**Pressure:** {weather['pressure']} hPa")

        st.markdown("---")
        st.subheader("📈 Temperature Forecast (Next 5 Hours)")
        for model, values in future_temp.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{model} - Chart**")
                st.line_chart(pd.DataFrame({model: values}, index=time_labels))
            with col2:
                df_display = pd.DataFrame({"Time": time_labels, "Temperature (°C)": [round(v, 1) for v in values]})
                st.write(f"**{model} - Values**")
                st.dataframe(df_display.set_index("Time"))

        st.markdown("---")
        st.subheader("💧 Humidity Forecast (Next 5 Hours)")
        for model, values in future_hum.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{model} - Chart**")
                st.line_chart(pd.DataFrame({model: values}, index=time_labels))
            with col2:
                df_display = pd.DataFrame({"Time": time_labels, "Humidity (%)": [round(v, 1) for v in values]})
                st.write(f"**{model} - Values**")
                st.dataframe(df_display.set_index("Time"))

    except FileNotFoundError:
        st.error("🚫 File not found. Please check the path and try again.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Please enter a city name and a valid file path to start.")
