"""Pronóstico de la demanda eléctrica horaria de Bangladesh con 24 h de anticipación.

Prepara los datos del PGCB Hourly Generation Dataset (UCI) y guarda el periodo de estudio con sus features en data/study_period.csv.

Uso: python train_model.py
"""
import os
import zipfile
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "", "pgcb+hourly+generation+dataset+(bangladesh).zip")
TARGET = "demand_mw"
# Límites físicos fijos (la capacidad de la red es < 17,000 MW); no se calculan con los datos, así que no hay fuga
MAX_MW = {
    "generation_mw": 20000, "demand_mw": 20000, "load_shedding": 5000,
    "gas": 15000, "liquid_fuel": 10000, "coal": 10000, "hydro": 1000,
    "india_bheramara_hvdc": 2000, "india_tripura": 500, "india_adani": 3000,
}
# Rezagos de la demanda: misma hora del día anterior y de la semana anterior (disponibles 24 h antes)
LAGS = {"demand_lag_24h": 24, "demand_lag_168h": 168}
STUDY_YEARS = 3
# Columnas de contexto: se guardan para el análisis, pero nunca son predictores (se miden en la misma hora)
CONTEXT = ["generation_mw", "load_shedding"]
WEATHER_PATH = os.path.join(HERE, "bangladesh_weather_hourly.csv")
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "cloud_cover"]
TIME_FEATURES = ["Hour", "Hour_sin", "Hour_cos", "DayOfWeek", "Month", "DayOfYear", "IsWeekend"]
FEATURES = ["datetime", *LAGS, *TIME_FEATURES, *WEATHER_VARS]
RANDOM_STATE = 42
NUMERIC_FEATURES = [*LAGS, "Hour_sin", "Hour_cos", "DayOfYear", "IsWeekend", *WEATHER_VARS]
ONEHOT_FEATURES = ["Hour", "Month", "DayOfWeek"]


def load_raw():
    """Lee el Excel del zip de UCI (92,650 registros horarios, abril de 2015 a junio de 2025)."""
    with zipfile.ZipFile(ZIP_PATH) as z:
        return pd.read_excel(z.open("PGCB_date_power_demand.xlsx"))


def clean(X):
    """Elimina duplicados y marcas de tiempo repetidas, ordena y descarta valores físicamente imposibles."""
    X = X.drop_duplicates().reset_index(drop=True)
    X = X.sort_values("datetime").drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)
    invalid = np.zeros(len(X), dtype=bool)
    for col, limit in MAX_MW.items():
        invalid |= X[col] > limit
    return X[~invalid].reset_index(drop=True)


def hourly_series(X):
    """Serie horaria de la demanda sobre todo el histórico limpio; los rezagos solo leen valores del pasado."""
    return X.groupby(X["datetime"].dt.floor("h"))[TARGET].mean()


def add_lag_features(df, series):
    df = df.copy()
    hour = df["datetime"].dt.floor("h")
    for name, hours in LAGS.items():
        df[name] = series.reindex(hour - pd.Timedelta(hours=hours)).to_numpy()
    return df


def add_time_features(df):
    df = df.copy()
    df["Hour"] = df["datetime"].dt.hour
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["DayOfWeek"] = df["datetime"].dt.dayofweek
    df["Month"] = df["datetime"].dt.month
    df["DayOfYear"] = df["datetime"].dt.dayofyear
    # En Bangladesh el fin de semana es viernes y sábado
    df["IsWeekend"] = df["DayOfWeek"].isin([4, 5]).astype(int)
    return df


def load_weather():
    return pd.read_csv(WEATHER_PATH, parse_dates=["time"], index_col="time")


def add_weather_features(df, weather):
    """Clima de la hora pronosticada (promedio de las 8 capitales); en operación vendría de un pronóstico."""
    df = df.copy()
    hourly_weather = weather.reindex(df["datetime"].dt.floor("h"))
    for var in WEATHER_VARS:
        df[var] = hourly_weather[var].to_numpy()
    return df


def build_features(X, series, weather):
    """Features disponibles el día anterior: rezagos de la demanda, calendario y clima."""
    features = add_lag_features(X[["datetime"]], series)
    return add_weather_features(add_time_features(features), weather)[FEATURES]


def make_preprocessor():
    """Imputación con la mediana y escalamiento para las numéricas; one-hot para hora, mes y día de la semana."""
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
         NUMERIC_FEATURES),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_FEATURES),
    ])


def report(name, y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name:<32} RMSE Test {rmse:8.2f} MW | MAPE {mape:5.2f}%")


def persistence(train_y, x):
    """Referencia: la demanda de la misma hora del día anterior (mediana de Train si falta)."""
    return x["demand_lag_24h"].fillna(train_y.median())


def prepare_data():
    """Limpia, calcula los rezagos con todo el histórico, recorta los últimos 3 años y divide 80/20 en orden."""
    X = clean(load_raw())
    series = hourly_series(X)
    X = X[X["datetime"] > X["datetime"].max() - pd.DateOffset(years=STUDY_YEARS)].reset_index(drop=True)
    X_var = build_features(X, series, load_weather())
    y_var = X[TARGET].copy()
    train_x, test_x, train_y, test_y = train_test_split(X_var, y_var, test_size=0.2, shuffle=False)
    return train_x, test_x, train_y, test_y, X[CONTEXT]


def main():
    train_x, test_x, train_y, test_y, context = prepare_data()
    report("persistencia 24 h", test_y, persistence(train_y, test_x))

    study = pd.concat([
        train_x.assign(**{TARGET: train_y, "split": "train"}),
        test_x.assign(**{TARGET: test_y, "split": "test"}),
    ]).reset_index(drop=True)
    study = pd.concat([study, context.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    study.to_csv(os.path.join(HERE, "data", "study_period.csv"), index=False)
    print("Guardado: data/study_period.csv")


if __name__ == "__main__":
    main()
