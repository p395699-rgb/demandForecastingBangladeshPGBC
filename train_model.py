"""Pronóstico de la demanda eléctrica horaria de Bangladesh con 24 h de anticipación.

Prepara los datos del PGCB Hourly Generation Dataset (UCI) y muestra un resumen de la limpieza.

Uso: python train_model.py
"""
import os
import zipfile
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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


def prepare_data():
    """Limpia, calcula los rezagos con todo el histórico, recorta los últimos 3 años y divide 80/20 en orden."""
    X = clean(load_raw())
    series = hourly_series(X)
    X = X[X["datetime"] > X["datetime"].max() - pd.DateOffset(years=STUDY_YEARS)].reset_index(drop=True)
    X_var = add_lag_features(X[["datetime"]], series)
    y_var = X[TARGET].copy()
    train_x, test_x, train_y, test_y = train_test_split(X_var, y_var, test_size=0.2, shuffle=False)
    return train_x, test_x, train_y, test_y, X[CONTEXT]


def main():
    train_x, test_x, train_y, test_y, context = prepare_data()
    print(f"Train: {len(train_x):,} | {train_x['datetime'].min()} → {train_x['datetime'].max()}")
    print(f"Test:  {len(test_x):,} | {test_x['datetime'].min()} → {test_x['datetime'].max()}")
    print("Rezagos faltantes:", pd.concat([train_x, test_x])[list(LAGS)].isna().sum().to_dict())


if __name__ == "__main__":
    main()
