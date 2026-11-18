# Demand Forecast — Bangladesh

Day-ahead (24 h) forecast of hourly electricity demand (MW) on Bangladesh's national grid with linear models, SVR and
ensembles (scikit-learn), served in a Streamlit app.

## Data

- [PGCB Hourly Generation Dataset](https://doi.org/10.24432/C59P6V) (UCI, CC BY 4.0), included as
  `pgcb+hourly+generation+dataset+(bangladesh).zip`.
- [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) hourly weather, mean of the 8 division capitals.
- [Shekh & Rafi (2026)](https://doi.org/10.17632/vpk8spw2mm.1) (Mendeley, CC BY 4.0): 6,342 later hours for
  out-of-sample validation.

## Method

- **Target:** `demand_mw`, 24 h ahead. Same-hour variables are excluded: in PGCB data demand ≈ generation + load
  shedding, so using them would be leakage.
- **Features:** demand lags (24 h, 168 h), calendar, and weather of the forecast hour (observed, as a perfect forecast).
- **Preprocessing:** median imputation, `StandardScaler` and one-hot (hour, month, weekday), fitted inside each fold.
- **Validation:** chronological 80/20 split (2022-06 → 2025-06), `TimeSeriesSplit(5, gap=24)` + `GridSearchCV`,
  24 h persistence as baseline.
- **Models:** Linear, Ridge, polynomial, SVR (linear, poly, RBF), AdaBoost, Gradient Boosting, Stacking, Voting.

Results and limitations are in the app's **Reporte técnico** tab; code structure is in `ARCHITECTURE.md`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

The `.joblib` artifacts require scikit-learn 1.7.2 (pinned). To regenerate them: `python train_model.py` (~4 min),
then `python prepare_new_data.py` and `python prepare_city_weather.py` (need internet).
