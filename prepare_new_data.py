"""Prepara las horas posteriores al periodo de estudio para validar los modelos con datos nunca vistos.

Fuente: "Hourly Electricity Generation, Demand, Load Shedding, and Fuel Mix Dataset for Bangladesh (2015–2026)",
Shekh y Rafi (2026), Mendeley Data, DOI 10.17632/vpk8spw2mm.1, CC BY 4.0. Mismas columnas que el dataset de UCI,
con datos hasta marzo de 2026. El clima se descarga del archivo histórico de Open-Meteo (8 capitales de división).

Uso: python prepare_new_data.py   (requiere internet si data/pgcb_V1_raw.csv no existe)
Genera: data/new_period.csv  (horas posteriores a data/study_period.csv con las mismas features y el objetivo)
"""
import os

import pandas as pd
import requests

from train_model import (CONTEXT, FEATURES, HERE, TARGET, WEATHER_VARS, add_lag_features, add_time_features,
                         add_weather_features, clean, hourly_series, load_raw)

MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/vpk8spw2mm/files/"
                "d0e7e374-f839-43cf-91d2-63c9f12b259e/file_downloaded")
RAW_PATH = os.path.join(HERE, "data", "pgcb_V1_raw.csv")
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CITIES = {
    "Dhaka": (23.8103, 90.4125), "Chattogram": (22.3569, 91.7832),
    "Khulna": (22.8456, 89.5403), "Rajshahi": (24.3745, 88.6042),
    "Sylhet": (24.8949, 91.8687), "Barishal": (22.7010, 90.3535),
    "Rangpur": (25.7439, 89.2752), "Mymensingh": (24.7471, 90.4203),
}


def download_raw():
    if not os.path.exists(RAW_PATH):
        response = requests.get(MENDELEY_URL, timeout=120)
        response.raise_for_status()
        with open(RAW_PATH, "wb") as f:
            f.write(response.content)
    return pd.read_csv(RAW_PATH, parse_dates=["datetime"])


def download_weather(start_date, end_date):
    params = {
        "latitude": ",".join(str(lat) for lat, _ in CITIES.values()),
        "longitude": ",".join(str(lon) for _, lon in CITIES.values()),
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS), "timezone": "Asia/Dhaka",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=120)
    response.raise_for_status()
    # Promedio nacional: media simple de las 8 ciudades en cada hora
    weather = pd.concat(pd.DataFrame(location["hourly"]) for location in response.json()).groupby("time").mean()
    weather.index = pd.to_datetime(weather.index)
    return weather


def main():
    study_end = pd.read_csv(os.path.join(HERE, "data", "study_period.csv"), usecols=["datetime"],
                            parse_dates=["datetime"])["datetime"].max()

    new = clean(download_raw())
    new = new[new["datetime"] > study_end].reset_index(drop=True)
    print("Registros nuevos:", len(new), "|", new["datetime"].min(), "→", new["datetime"].max())

    # Los rezagos se calculan con la serie completa (histórico de UCI + horas nuevas), siempre con valores del pasado
    series = pd.concat([hourly_series(clean(load_raw())), hourly_series(new)]).groupby(level=0).last()
    weather = download_weather(f"{new['datetime'].min():%Y-%m-%d}", f"{new['datetime'].max():%Y-%m-%d}")
    features = add_weather_features(add_time_features(add_lag_features(new[["datetime"]], series)), weather)[FEATURES]
    print("Horas sin clima:", features["temperature_2m"].isna().sum(),
          "| sin rezago de 24 h:", features["demand_lag_24h"].isna().sum())

    new_period = pd.concat([features, new[[TARGET, *CONTEXT]]], axis=1)
    new_period.to_csv(os.path.join(HERE, "data", "new_period.csv"), index=False)
    print("Guardado: data/new_period.csv")


if __name__ == "__main__":
    main()
