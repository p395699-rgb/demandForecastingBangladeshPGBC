"""Descarga el clima horario de cada una de las 8 capitales de división (Open-Meteo) para el periodo de estudio.

Las variables exógenas del modelo son el promedio nacional de estas 8 ciudades (bangladesh_weather_hourly.csv). Este
archivo conserva el detalle por ciudad para el mapa de correlaciones de la app; los modelos no lo usan.

Uso: python prepare_city_weather.py
Genera: data/weather_by_city.csv (time, city, latitude, longitude y las 5 variables de clima)
"""
import os

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "cloud_cover"]
# Mismas coordenadas que en los notebooks y en prepare_new_data.py
CITIES = {
    "Dhaka": (23.8103, 90.4125), "Chattogram": (22.3569, 91.7832),
    "Khulna": (22.8456, 89.5403), "Rajshahi": (24.3745, 88.6042),
    "Sylhet": (24.8949, 91.8687), "Barishal": (22.7010, 90.3535),
    "Rangpur": (25.7439, 89.2752), "Mymensingh": (24.7471, 90.4203),
}


def main():
    study = pd.read_csv(os.path.join(HERE, "data", "study_period.csv"), usecols=["datetime"], parse_dates=["datetime"])
    params = {
        "latitude": ",".join(str(lat) for lat, _ in CITIES.values()),
        "longitude": ",".join(str(lon) for _, lon in CITIES.values()),
        "start_date": f"{study['datetime'].min():%Y-%m-%d}", "end_date": f"{study['datetime'].max():%Y-%m-%d}",
        "hourly": ",".join(WEATHER_VARS), "timezone": "Asia/Dhaka",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=180)
    response.raise_for_status()

    frames = []
    for (city, (lat, lon)), location in zip(CITIES.items(), response.json()):
        frame = pd.DataFrame(location["hourly"])
        frames.append(frame.assign(city=city, latitude=lat, longitude=lon))
    weather = pd.concat(frames, ignore_index=True)
    weather = weather[["time", "city", "latitude", "longitude", *WEATHER_VARS]]
    weather.to_csv(os.path.join(HERE, "data", "weather_by_city.csv"), index=False)
    print(f"{len(weather):,} registros | {weather['time'].min()} → {weather['time'].max()}")


if __name__ == "__main__":
    main()
