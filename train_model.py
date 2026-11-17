"""Pronóstico de la demanda eléctrica horaria de Bangladesh con 24 h de anticipación.

Prepara los datos del PGCB Hourly Generation Dataset (UCI) y muestra un resumen de la limpieza.

Uso: python train_model.py
"""
import os
import zipfile
import warnings

import numpy as np
import pandas as pd

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


def main():
    raw = load_raw()
    X = clean(raw)
    print(f"Registros: {len(raw):,} → {len(X):,} tras la limpieza | {X['datetime'].min()} → {X['datetime'].max()}")
    identity = (X[TARGET] - X["generation_mw"] - X["load_shedding"]).abs() <= 1
    print(f"demand_mw = generation_mw + load_shedding (±1 MW) en {identity.mean():.1%} de los registros")


if __name__ == "__main__":
    main()
