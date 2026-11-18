# Arquitectura

El proyecto separa dos momentos:

1. **Fuera de línea (scripts):** se preparan los datos, se entrenan los modelos y se guardan como artefactos en
   `data/` y `model/`.
2. **En línea (app):** `app.py` solo **carga** esos artefactos y predice. Nunca entrena. Los resultados de validación
   cruzada y de las búsquedas en rejilla vienen de `LoadSheddingDemandaAllModels.ipynb` y viven como constantes en
   `app.py`.

```
DemandForecast/
├── app.py                         # App de Streamlit: dos pestañas (producto y reporte técnico)
├── components.py                  # Componentes de interfaz reutilizables (tema, CSS, tablas, tarjetas, gráficas)
├── references.py                  # Referencias del reporte (mismas claves que references.bib)
├── train_model.py                 # Limpieza, rezagos, features, partición y entrenamiento de todos los modelos
├── prepare_new_data.py            # Horas posteriores al estudio (Mendeley + Open-Meteo) con las mismas features
├── prepare_city_weather.py        # Clima por capital, solo para el mapa del EDA
├── bangladesh_weather_hourly.csv  # Clima horario promedio de las 8 capitales (entrada de train_model.py)
├── pgcb+hourly+generation+dataset+(bangladesh).zip  # Dataset de UCI (entrada de train_model.py)
├── requirements.txt
├── .streamlit/config.toml         # Tema (colores claro/oscuro, fuentes IBM Plex)
├── data/
│   ├── study_period.csv           # Periodo de estudio: features, objetivo, split y columnas de contexto
│   ├── new_period.csv             # Horas nunca vistas para validar los modelos
│   └── weather_by_city.csv        # Clima por capital (mapa)
└── model/                         # Artefactos joblib
```

## `train_model.py`

| Función | Responsabilidad |
|---|---|
| `load_raw()`, `clean()` | Lee el zip de UCI, elimina duplicados y valores imposibles (`MAX_MW`). |
| `hourly_series()`, `add_lag_features()` | Rezagos de 24 y 168 h con todo el histórico, solo con valores del pasado. |
| `add_time_features()`, `add_weather_features()` | Calendario y clima de la hora pronosticada. |
| `build_features()`, `prepare_data()` | Arma `FEATURES`, recorta 3 años y divide 80/20 en orden. |
| `make_preprocessor()` | Mediana + `StandardScaler` para numéricas; one-hot para hora, mes y día de la semana. |
| `train_linear()`, `train_svr()`, `train_boosting()`, `train_stacking()`, `train_voting()` | Entrenan y guardan cada familia. |

Los hiperparámetros son constantes (`BEST_*`) elegidas con `GridSearchCV` en el notebook. `generation_mw` y
`load_shedding` se guardan en `study_period.csv` como **contexto** para el análisis, pero nunca son predictores.

## `app.py`

- **Carga cacheada:** `load_artifact()` (`@st.cache_resource`) y `load_data()` / `load_new_data()`
  (`@st.cache_data`), que agregan una columna `prediction_*` por modelo con `predict_all()`.
- **Pestaña Predicción:** contexto, validación con horas nuevas, día de mayor demanda, casos al azar, simulador
  (arma una fila con `add_time_features()` y las features de `FEATURES`) y alcance.
- **Pestaña Reporte técnico:** 18 secciones definidas en `SECTIONS`; `heading()` las numera y la barra lateral
  muestra el índice.
- **Helpers:** `callout()`, `caption()`, `table()`, `metrics()`, `series_chart()`, `bar_chart()`, `fold_chart()` y
  `residuals_by_hour()`.

## Qué regenerar cuando algo cambia

| Cambio | Ejecutar | Actualizar a mano en `app.py` |
|---|---|---|
| Limpieza, rezagos o features | `train_model.py`, luego los dos `prepare_*` | Textos y constantes con conteos o métricas |
| Hiperparámetros | `train_model.py` | Constantes de CV del modelo |
| Versión de scikit-learn | `train_model.py` | — |
