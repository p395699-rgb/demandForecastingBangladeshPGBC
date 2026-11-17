"""App de Streamlit: pronóstico de la demanda eléctrica horaria de Bangladesh con 24 h de anticipación.

Solo carga los artefactos de model/ y data/ (generados por train_model.py y los scripts prepare_*.py) y predice.
Uso: streamlit run app.py
"""
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from train_model import FEATURES, add_time_features

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "demand_mw"
MENDELEY_URL = "https://doi.org/10.17632/vpk8spw2mm.1"

st.set_page_config(page_title="Demanda eléctrica de Bangladesh — pronóstico a 24 h", page_icon="⚡", layout="centered")

# Tokens de notebook-theme-preview.html; los colores base vienen de .streamlit/config.toml
LIGHT = {
    "accent": "#5B3DC8", "accent-soft": "#ECE6FC", "mark": "#E3D9FF", "surface": "#F5F2FA",
    "border": "#E4DFEE", "fg": "#1F1B2D", "muted": "#686179", "tip": "#127F6E", "warn": "#A86A00",
}
DARK = {
    "accent": "#B39DFF", "accent-soft": "#2A2245", "mark": "#43357A", "surface": "#1E1A29",
    "border": "#2F2940", "fg": "#ECE8F5", "muted": "#A39BB7", "tip": "#4FCBB5", "warn": "#E3A94A",
}
tokens = DARK if st.context.theme.type == "dark" else LIGHT
css_vars = "".join(f"--nb-{k}: {v};" for k, v in tokens.items())

st.html(f"""
<style>
:root {{ {css_vars}
  --nb-font-display: "IBM Plex Mono", "Courier New", monospace; }}
.block-container {{ padding-top: 3.5rem; }}
.nb-title {{ font-family: var(--nb-font-display); font-size: clamp(2rem, 5vw, 2.9rem); font-weight: 600;
  line-height: 1.12; letter-spacing: -0.045em; word-spacing: -0.2em; margin: 0 0 1.1rem; text-wrap: balance; }}
.nb-description {{ font-size: 1.2rem; line-height: 1.5; color: var(--nb-muted); max-width: 52ch; margin: 0 0 1.5rem; }}
.nb-meta {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; padding-top: 1rem; margin: 0;
  border-top: 1px solid var(--nb-border); color: var(--nb-muted); font-size: 0.8rem; font-family: var(--nb-font-display); }}
.nb-tags {{ display: flex; flex-wrap: wrap; gap: 0.4rem; list-style: none; padding: 0 !important; margin: 1rem 0 0 !important; }}
.nb-tags li {{ font-family: var(--nb-font-display); font-size: 0.75rem; font-weight: 500; padding: 0.15rem 0.6rem;
  border-radius: 999px; background: var(--nb-accent-soft); color: var(--nb-accent); margin: 0; }}
.nb-h2 {{ font-family: var(--nb-font-display); font-size: 1.45rem; font-weight: 600; letter-spacing: -0.035em;
  word-spacing: -0.15em; margin: 2.75rem 0 0.25rem; scroll-margin-top: 4rem; color: var(--nb-fg); }}
.nb-h2 span {{ color: var(--nb-accent); margin-right: 0.6em; }}
.nb-callout {{ --kind: var(--nb-accent); margin: 1rem 0 1.5rem; padding: 0.9rem 1.15rem 0.9rem 1.1rem;
  border-left: 3px solid var(--kind); border-radius: 0 4px 4px 0; background: var(--nb-surface); }}
.nb-callout[data-kind="result"] {{ background: var(--nb-accent-soft); }}
.nb-callout[data-kind="tip"] {{ --kind: var(--nb-tip); }}
.nb-callout[data-kind="warning"] {{ --kind: var(--nb-warn); }}
.nb-callout-label {{ display: block; font-family: var(--nb-font-display); font-size: 0.75rem; font-weight: 600;
  color: var(--kind); margin-bottom: 0.25rem; }}
.nb-callout p {{ margin: 0; }}
mark {{ background: var(--nb-mark); color: inherit; padding: 0.05em 0.25em; border-radius: 2px; }}
.nb-caption {{ font-size: 0.875rem; line-height: 1.5; color: var(--nb-muted); margin: 0.25rem 0 1.5rem; max-width: 60ch; }}
.nb-caption b {{ font-family: var(--nb-font-display); font-size: 0.92em; color: var(--nb-fg); }}
.nb-table {{ border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; line-height: 1.45;
  border-top: 1.5px solid var(--nb-fg); border-bottom: 1.5px solid var(--nb-fg); width: 100%; margin: 0.5rem 0; }}
.nb-table th {{ text-align: right; font-weight: 600; font-family: var(--nb-font-display); font-size: 0.8rem;
  padding: 0.55rem 0.8rem 0.45rem; border-bottom: 1px solid var(--nb-fg); }}
.nb-table td {{ text-align: right; padding: 0.4rem 0.8rem; border-bottom: 1px solid var(--nb-border); }}
.nb-table th:first-child, .nb-table td:first-child {{ text-align: left; }}
.nb-table tr:last-child td {{ border-bottom: 0; }}
.nb-table tr.best td {{ font-weight: 600; }}
.nb-toc-title {{ font-family: var(--nb-font-display); font-weight: 600; font-size: 0.85rem; margin-bottom: 0.75rem; }}
.nb-toc ol {{ list-style: none; padding: 0; margin: 0; counter-reset: toc; font-family: var(--nb-font-display); font-size: 0.8rem; }}
.nb-toc li {{ counter-increment: toc; padding: 0.3rem 0 0.3rem 0.85rem; border-left: 2px solid var(--nb-border); margin: 0; }}
.nb-toc li:hover {{ border-left-color: var(--nb-accent); }}
.nb-toc a {{ color: var(--nb-muted) !important; text-decoration: none !important; }}
.nb-toc a::before {{ content: counter(toc) ". "; }}
.nb-toc a:hover {{ color: var(--nb-accent) !important; }}
.nb-refs {{ padding-left: 1.2rem !important; margin: 0.5rem 0 0 !important; line-height: 1.6; }}
.nb-refs li {{ margin-bottom: 0.6rem; }}
.nb-refs a {{ color: var(--nb-accent) !important; text-decoration: underline; text-underline-offset: 3px;
  text-decoration-color: color-mix(in srgb, var(--nb-accent) 40%, transparent); }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ font-family: var(--nb-font-display); font-size: 0.78rem; color: var(--nb-muted); }}
.pd-hero {{ padding: 0.5rem 0 0.5rem; }}
.pd-kicker {{ font-family: var(--nb-font-display); font-size: 0.8rem; font-weight: 600; color: var(--nb-accent); margin: 0 0 0.6rem; }}
.pd-title {{ font-family: var(--nb-font-display); font-size: clamp(2rem, 5.5vw, 3rem); font-weight: 600; line-height: 1.1;
  letter-spacing: -0.045em; word-spacing: -0.2em; margin: 0 0 1rem; text-wrap: balance; }}
.pd-lead {{ font-size: 1.15rem; line-height: 1.6; color: var(--nb-muted); max-width: 56ch; margin: 0; }}
.pd-h2 {{ font-family: var(--nb-font-display); font-size: 1.4rem; font-weight: 600; letter-spacing: -0.035em;
  word-spacing: -0.15em; margin: 2.5rem 0 0.5rem; color: var(--nb-fg); text-wrap: balance; }}
.pd-h2 small {{ display: block; font-size: 0.72rem; letter-spacing: 0.08em; word-spacing: 0; text-transform: uppercase;
  color: var(--nb-accent); margin-bottom: 0.35rem; }}
.pd-text {{ line-height: 1.6; margin: 0 0 0.75rem; }}
.pd-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.75rem; margin: 1rem 0; }}
.pd-step {{ background: var(--nb-surface); border: 1px solid var(--nb-border); border-radius: 6px; padding: 1rem; }}
.pd-step b {{ display: block; font-family: var(--nb-font-display); font-size: 0.9rem; margin-bottom: 0.35rem; }}
.pd-step span {{ font-size: 0.9rem; line-height: 1.5; color: var(--nb-muted); }}
.pd-result {{ background: var(--nb-accent-soft); border-left: 3px solid var(--nb-accent); border-radius: 0 6px 6px 0;
  padding: 1.1rem 1.3rem; margin: 1.25rem 0 0.5rem; }}
.pd-result p {{ margin: 0.6rem 0 0; line-height: 1.5; }}
.pd-big {{ font-family: var(--nb-font-display); font-size: clamp(2.4rem, 7vw, 3.4rem); font-weight: 600;
  letter-spacing: -0.04em; line-height: 1; color: var(--nb-accent); font-variant-numeric: tabular-nums; }}
.pd-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin: 1rem 0 1.25rem; }}
.pd-stat {{ border-top: 3px solid var(--nb-accent); background: var(--nb-surface); border-radius: 0 0 6px 6px; padding: 0.9rem 1rem; }}
.pd-stat > b {{ display: block; font-family: var(--nb-font-display); font-size: 1.6rem; font-weight: 600; letter-spacing: -0.03em;
  color: var(--nb-accent); font-variant-numeric: tabular-nums; margin-bottom: 0.3rem; }}
.pd-stat span {{ font-size: 0.88rem; line-height: 1.5; color: var(--nb-muted); }}
.pd-list {{ padding-left: 1.2rem !important; line-height: 1.6; }}
.pd-list li {{ margin-bottom: 0.5rem; }}
</style>
""")

SECTIONS = [
    ("introduccion", "Introducción"),
    ("inspeccion", "Inspección inicial"),
    ("objetivo", "Variable objetivo y fuga de información"),
    ("particion", "Limpieza, rezagos y partición"),
    ("eda", "Análisis exploratorio (Train)"),
    ("pipeline", "Preprocesamiento"),
    ("cv", "Validación cruzada y persistencia"),
    ("lineales", "Modelos lineales y polinomial"),
    ("kernels", "Comparación de kernels"),
    ("optimizacion", "Optimización de C y γ"),
    ("test", "Evaluación final en Test"),
    ("ensambles", "Ensambles: metodología"),
    ("ensambles_cv", "Ensambles: validación cruzada"),
    ("ensambles_test", "Ensambles: Test"),
    ("nuevos", "Validación con datos nuevos"),
    ("discusion", "Discusión y estado del arte"),
    ("conclusiones", "Conclusiones"),
    ("referencias", "Referencias"),
]
LABELS = {
    TARGET: "Real", "prediction": "SVR RBF", "prediction_persistence": "Persistencia 24 h",
    "prediction_stacking": "Stacking", "prediction_linear": "Regresión lineal", "prediction_ridge": "Ridge",
    "prediction_poly": "Polinomial", "prediction_svr_linear": "SVR lineal", "prediction_gb": "Gradient Boosting",
    "prediction_ada": "AdaBoost", "prediction_voting_simple": "Voting simple", "prediction_voting": "Voting ponderado",
    "prediction_voting_hard": "Voting hard",
}
SERIES = list(LABELS.values())
figure_count = 0


def heading(anchor):
    number = [a for a, _ in SECTIONS].index(anchor) + 1
    st.html(f'<h2 class="nb-h2" id="{anchor}"><span>{number}</span>{dict(SECTIONS)[anchor]}</h2>')


def callout(kind, label, text):
    st.html(f'<div class="nb-callout" data-kind="{kind}"><span class="nb-callout-label">{label}</span><p>{text}</p></div>')


def caption(text):
    global figure_count
    figure_count += 1
    st.html(f'<p class="nb-caption"><b>Figura {figure_count}.</b> {text}</p>')


def table(header, rows, best=None, note=None):
    def cell(value):
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else value

    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        f'<tr class="{"best" if i == best else ""}">' + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>"
        for i, row in enumerate(rows)
    )
    st.html(f'<div style="overflow-x:auto"><table class="nb-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    if note:
        st.html(f'<p class="nb-caption">{note}</p>')


def metrics(y_true, y_pred):
    """RMSE, MAE y MAPE en MW y %, y R²; el MAPE es interpretable porque la demanda nunca es cero."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def series_chart(frame, y_title, height=320):
    """Serie temporal: el valor real en línea gruesa, los pronósticos en líneas finas y la persistencia punteada."""
    frame = frame.rename(columns=LABELS)
    present = [s for s in SERIES if s in frame.columns]
    long = frame.reset_index().melt(id_vars="datetime", var_name="Serie", value_name="MW")
    base = alt.Chart(long).encode(
        x=alt.X("datetime:T", title=None),
        y=alt.Y("MW:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("Serie:N", sort=present, scale=alt.Scale(domain=present),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("datetime:T", title="Fecha"), "Serie:N", alt.Tooltip("MW:Q", format=",.0f")],
    )
    real = base.transform_filter(alt.datum.Serie == "Real").mark_line(strokeWidth=2.4)
    pred = base.transform_filter(alt.datum.Serie != "Real").mark_line(strokeWidth=1.3).encode(
        strokeDash=alt.condition(alt.datum.Serie == "Persistencia 24 h", alt.value([4, 3]), alt.value([1, 0])))
    st.altair_chart(alt.layer(real, pred).properties(height=height).interactive(bind_y=False), width="stretch")


def bar_chart(frame, x, y, x_title, y_title, height=240, zero=True):
    chart = alt.Chart(frame).mark_bar().encode(
        x=alt.X(f"{x}:O", title=x_title, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=zero)),
        tooltip=[x, alt.Tooltip(f"{y}:Q", format=",.1f")],
    )
    st.altair_chart(chart.properties(height=height), width="stretch")


def fold_chart(folds, height=280):
    """RMSE de validación por fold: barras agrupadas por modelo."""
    long = (pd.DataFrame(folds, index=[f"Fold {i}" for i in range(1, 6)]).reset_index(names="Fold")
            .melt(id_vars="Fold", var_name="Modelo", value_name="RMSE"))
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            x=alt.X("Fold:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="Modelo:N",
            y=alt.Y("RMSE:Q", title="RMSE de validación (MW)"),
            color=alt.Color("Modelo:N", sort=list(folds), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Fold", "Modelo", alt.Tooltip("RMSE:Q", format=",.1f")],
        ).properties(height=height),
        width="stretch",
    )


def residuals_by_hour(frame, column):
    """Residuo medio (real − pronóstico) y RMSE por hora del día."""
    error = frame[TARGET] - frame[column]
    hours = frame["datetime"].dt.hour
    return pd.DataFrame({"Residuo medio": error.groupby(hours).mean(),
                         "RMSE": error.groupby(hours).apply(lambda e: np.sqrt((e ** 2).mean()))})


@st.cache_resource
def load_artifact(name):
    return joblib.load(os.path.join(HERE, "model", name))


def predict_all(df, train_median):
    """Agrega una columna por modelo. Los modelos base del Voting dan también las predicciones de GB y AdaBoost."""
    features = df[FEATURES]
    linear, voting = load_artifact("linear.joblib"), load_artifact("voting.joblib")
    base = voting["simple"].transform(features)  # columnas: SVR RBF, Gradient Boosting, AdaBoost
    return df.assign(
        prediction=load_artifact("svr_rbf.joblib")["model"].predict(features),
        prediction_persistence=df["demand_lag_24h"].fillna(train_median),
        prediction_stacking=load_artifact("stacking.joblib")["model"].predict(features),
        prediction_linear=linear["linear"].predict(features),
        prediction_ridge=linear["ridge"].predict(features),
        prediction_poly=linear["poly"].predict(features),
        prediction_svr_linear=load_artifact("svr_linear.joblib")["model"].predict(features),
        prediction_gb=base[:, 1],
        prediction_ada=base[:, 2],
        prediction_voting_simple=base.mean(axis=1),
        prediction_voting=voting["weighted"].predict(features),
        # Hard voting en regresión: mediana de las predicciones de los tres modelos base
        prediction_voting_hard=np.median(base, axis=1),
    )


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(HERE, "data", "study_period.csv"), parse_dates=["datetime"])
    train_median = float(df.loc[df["split"] == "train", TARGET].median())
    return predict_all(df, train_median), train_median


@st.cache_data
def load_new_data(train_median):
    # Horas posteriores al periodo de estudio (jun 2025 – mar 2026); se generan con prepare_new_data.py
    df = pd.read_csv(os.path.join(HERE, "data", "new_period.csv"), parse_dates=["datetime"])
    return predict_all(df, train_median)


MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre"]
DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def month_year(ts):
    return f"{MONTHS[ts.month - 1]} de {ts.year}"


svr_artifact = load_artifact("svr_rbf.joblib")
model, params = svr_artifact["model"], svr_artifact["params"]
stacking = load_artifact("stacking.joblib")["model"]
linear_artifact = load_artifact("linear.joblib")
voting_artifact = load_artifact("voting.joblib")
df, train_median = load_data()
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
y_true = test[TARGET]
svr_test = metrics(y_true, test["prediction"])
persistence_test = metrics(y_true, test["prediction_persistence"])

with st.sidebar:
    items = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in SECTIONS)
    st.html(f'<div class="nb-toc-title">Reporte técnico</div><nav class="nb-toc"><ol>{items}</ol></nav>')

tab_product, tab_report = st.tabs(["Predicción", "Reporte técnico"])

with tab_report:
    st.html("""
    <header>
      <h1 class="nb-title">Pronosticando la demanda eléctrica de Bangladesh con un día de anticipación</h1>
      <p class="nb-description">Modelos lineales, Support Vector Regression y métodos de ensamble sobre la demanda de días
      anteriores, el calendario y el clima para pronosticar los MW que requerirá la red nacional en cada hora.</p>
      <p class="nb-meta"><span>Andrea Rodríguez · 395699</span><span>Universidad Autónoma de Chihuahua</span><span>Machine Learning</span></p>
      <ul class="nb-tags"><li>machine learning</li><li>series de tiempo</li><li>SVR</li><li>ensambles</li><li>scikit-learn</li></ul>
    </header>
    """)

    heading("introduccion")
    st.markdown(
        "Cuando la demanda supera a la generación disponible, la Power Grid Company of Bangladesh (PGCB) desconecta "
        "carga de forma programada: el **load shedding**. Anticipar la demanda del día siguiente es el insumo básico para "
        "planificar qué centrales despachar, cuánto combustible e importaciones asegurar y cómo distribuir los cortes. "
        "El objetivo es pronosticar la demanda de cada hora (`demand_mw`, MW) **con 24 h de anticipación**, usando solo "
        "información disponible el día anterior: un problema de **regresión** supervisada.\n\n"
        "Flujo: Datos → Inspección → Variable objetivo y fuga → Rezagos y partición → EDA solo en Train → Pipeline → "
        "Validación cruzada con persistencia → Modelos lineales → SVR → Ensambles → Test → Datos nuevos."
    )
    callout(
        "result", "Resultado",
        f"El SVR RBF optimizado obtiene en Test un MAPE de <mark>{svr_test['mape']:.2f}%</mark> (RMSE de "
        f"{svr_test['rmse']:,.0f} MW), frente a {persistence_test['mape']:.2f}% de la persistencia de 24 h: "
        f"{1 - svr_test['rmse'] / persistence_test['rmse']:.0%} menos error. También fue el mejor en validación "
        "cruzada (RMSE de 940.61 MW) y mantiene su desempeño en datos posteriores al periodo de estudio.",
    )

    heading("inspeccion")
    RAW_MISSING = [("solar", 22133, 23.9), ("wind", 73974, 79.8), ("india_adani", 85312, 92.1),
                   ("nepal", 87299, 94.2), ("remarks", 86257, 93.1)]
    st.markdown(
        "- **Dataset:** PGCB Hourly Generation Dataset (UCI): 92,650 registros horarios y 15 columnas, de 2015-04 a "
        "2025-06. Se complementa con clima horario de Open-Meteo (promedio de las 8 capitales de división).\n"
        "- **Objetivo:** `demand_mw`, sin faltantes.\n"
        "- **Duplicados:** 160 registros completos y 432 marcas de tiempo repetidas.\n"
        "- **Valores imposibles:** `demand_mw` = 156,050 MW o `generation_mw` = 64,526,500 MW, muy por encima de la "
        "capacidad de la red (< 17,000 MW)."
    )
    table(["Variable", "Faltantes", "%"], RAW_MISSING,
          note="Valores faltantes en el histórico completo: fuentes que empezaron a registrarse tarde. Ninguna es "
               "predictor del modelo de demanda.")

    heading("objetivo")
    with_shedding = df[df["load_shedding"] >= 50]
    ratio = (with_shedding[TARGET] - with_shedding["generation_mw"]) / with_shedding["load_shedding"]
    ratio_by_year = ratio.groupby(with_shedding["datetime"].dt.year).median()
    st.markdown(
        "La demanda de PGCB **no es una medición independiente**. En el 84.8% de los registros del histórico completo "
        "se cumple `demand_mw = generation_mw + load_shedding` (±1 MW). En las horas con cortes, desde 2023 la "
        "diferencia entre demanda y generación equivale a un factor casi constante sobre el load shedding:"
    )
    table(["Año", "(demanda − generación) / load shedding", "Horas"],
          [(str(year), f"{value:.3f}", int((with_shedding["datetime"].dt.year == year).sum()))
           for year, value in ratio_by_year.items()],
          note="Mediana en las horas del periodo de estudio con al menos 50 MW de load shedding.")
    callout(
        "warning", "Fuga de información",
        "Con la generación o el load shedding de la misma hora, un modelo solo tendría que sumarlos. Por eso se "
        "excluyen todas las variables medidas en la misma hora: generación, load shedding, generación por fuente, "
        "importaciones y <code>remarks</code>. Solo se usa información disponible el día anterior.",
    )
    callout(
        "tip", "¿Por qué la demanda y no el load shedding?",
        "Primero se intentó predecir el load shedding. Con variables de la misma hora, el modelo solo reconstruía la "
        "identidad contable (<i>nowcasting</i>); con 24 h de anticipación, el mejor modelo apenas mejoró a la persistencia "
        "en validación (279 frente a 294 MW) y no la superó en prueba. La demanda, en cambio, sigue patrones de "
        "calendario y clima que sí pueden anticiparse, y es la variable que reporta la literatura.",
    )

    heading("particion")
    st.markdown(
        "Reglas fijas antes de la partición (no se calculan con los datos, así que no generan fuga de información):\n"
        "- Eliminación de duplicados y marcas de tiempo repetidas, y orden cronológico.\n"
        "- Eliminación de 168 registros con valores físicamente imposibles (límites de capacidad por columna).\n"
        "- **Rezagos:** `demand_lag_24h` (misma hora del día anterior) y `demand_lag_168h` (misma hora de la semana "
        "anterior), calculados sobre la serie horaria completa y solo con valores del pasado.\n"
        "- **Periodo de estudio:** últimos 3 años, que reflejan el nivel de demanda y la operación más recientes.\n\n"
        f"Partición **cronológica** 80/20 (`shuffle=False`): Train con {len(train):,} registros "
        f"({train['datetime'].min():%Y-%m-%d} → {train['datetime'].max():%Y-%m-%d}) y Test con {len(test):,} "
        f"({test['datetime'].min():%Y-%m-%d} → {test['datetime'].max():%Y-%m-%d})."
    )
    context = (
        df.set_index("datetime")[[TARGET, "generation_mw"]].resample("D").mean()
        .rename(columns={TARGET: "Demanda", "generation_mw": "Generación"})
        .reset_index().melt(id_vars="datetime", var_name="Serie", value_name="MW")
    )
    lines = alt.Chart(context).mark_line(strokeWidth=1).encode(
        x=alt.X("datetime:T", title=None), y=alt.Y("MW:Q", title="MW (promedio diario)", scale=alt.Scale(zero=False)),
        color=alt.Color("Serie:N", legend=alt.Legend(orient="top", title=None)),
    )
    split_rule = alt.Chart(pd.DataFrame({"datetime": [test["datetime"].min()]})).mark_rule(strokeDash=[4, 4]).encode(x="datetime:T")
    st.altair_chart((lines + split_rule).properties(height=300), width="stretch")
    caption("Demanda y generación en el periodo de estudio (promedios diarios). La línea punteada marca el inicio de "
            "Test. La brecha entre ambas es el load shedding.")
    callout(
        "warning", "Tendencia y estacionalidad",
        "La demanda tiene una estacionalidad anual marcada (máximo en verano, mínimo en invierno) y una tendencia de "
        "crecimiento. Los modelos que no extrapolan, como los árboles, quedan en desventaja frente a esa tendencia.",
    )

    heading("eda")
    missing_lags = train[["demand_lag_24h", "demand_lag_168h"]].isna().sum()
    counts, edges = np.histogram(train[TARGET], bins=40)
    hist = pd.DataFrame({"MW": edges[:-1].round(0).astype(int), "Frecuencia": counts})
    st.altair_chart(
        alt.Chart(hist).mark_bar().encode(x=alt.X("MW:Q", title="demand_mw (MW)"), y="Frecuencia:Q").properties(height=220),
        width="stretch",
    )
    caption(
        f"Distribución de la demanda en Train: media de {train[TARGET].mean():,.1f} MW, desviación estándar de "
        f"{train[TARGET].std():,.1f} MW y mediana de {train[TARGET].median():,.1f} MW. Es aproximadamente simétrica y no "
        f"tiene ceros, por lo que el MAPE es interpretable. Hay {(train[TARGET] < 5000).sum()} horas con demanda menor "
        "de 5,000 MW (registros anómalos), ninguna en Test."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        by_hour = train.groupby("Hour")[TARGET].mean().reset_index()
        bar_chart(by_hour, "Hour", TARGET, "Hora del día", "MW promedio", zero=False)
    with col_b:
        by_month = train.groupby("Month")[TARGET].mean().reset_index()
        bar_chart(by_month, "Month", TARGET, "Mes", "MW promedio", zero=False)
    caption(f"Demanda promedio por hora y por mes (Train): mínimo a las {by_hour.loc[by_hour[TARGET].idxmin(), 'Hour']:02d}:00 "
            f"y máximo a las {by_hour.loc[by_hour[TARGET].idxmax(), 'Hour']:02d}:00; mínimo en "
            f"{MONTHS[by_month.loc[by_month[TARGET].idxmin(), 'Month'] - 1]} y máximo en "
            f"{MONTHS[by_month.loc[by_month[TARGET].idxmax(), 'Month'] - 1]}. Las relaciones no son monótonas, por lo que "
            "la hora, el mes y el día de la semana se codifican con one-hot.")

    sample = train.dropna(subset=["demand_lag_24h"]).sample(4000, random_state=42)
    st.altair_chart(
        alt.Chart(sample).mark_circle(size=8, opacity=0.3).encode(
            x=alt.X("demand_lag_24h:Q", title="Demanda 24 h antes (MW)", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{TARGET}:Q", title="Demanda (MW)", scale=alt.Scale(zero=False)),
        ).properties(height=260),
        width="stretch",
    )
    caption(f"Demanda frente a la demanda de la misma hora del día anterior (muestra de 4,000 horas de Train). Relación "
            "casi lineal: los puntos alejados de la diagonal son cambios bruscos entre días (feriados, lluvias) y "
            f"registros anómalos. Solo {int(missing_lags.max())} horas de Train no tienen rezago y se imputan con la mediana.")

    numeric = [c for c in FEATURES if c not in ("datetime", "Hour", "Month", "DayOfWeek")]
    corr = train[numeric].corrwith(train[TARGET]).sort_values(key=abs, ascending=False)
    corr_frame = corr.rename("r").reset_index().rename(columns={"index": "Variable"})
    st.altair_chart(
        alt.Chart(corr_frame).mark_bar().encode(
            x=alt.X("r:Q", title="Correlación de Pearson con demand_mw"),
            y=alt.Y("Variable:N", sort=None, title=None),
            tooltip=["Variable", alt.Tooltip("r:Q", format=".3f")],
        ).properties(height=320),
        width="stretch",
    )
    caption("Correlación con la demanda en Train. El rezago de 24 h es el predictor más informativo, seguido de la "
            "sensación térmica: el uso de ventiladores y aire acondicionado depende del calor percibido. No se aplica un "
            "umbral de correlación, porque la hora y el día de la semana tienen relaciones no monótonas.")

    WEATHER_LABELS = {"apparent_temperature": "Sensación térmica", "temperature_2m": "Temperatura",
                      "relative_humidity_2m": "Humedad relativa", "cloud_cover": "Nubosidad", "precipitation": "Precipitación"}

    @st.cache_data
    def load_city_correlations(train_hours, train_target):
        """Correlación de Pearson entre el clima de cada capital y la demanda en Train (prepare_city_weather.py)."""
        weather = pd.read_csv(os.path.join(HERE, "data", "weather_by_city.csv"), parse_dates=["time"])
        target = pd.Series(train_target.to_numpy(), index=train_hours.dt.floor("h").to_numpy(), name=TARGET)
        rows = []
        for (city, lat, lon), group in weather.groupby(["city", "latitude", "longitude"]):
            merged = group.set_index("time").join(target, how="inner")
            rows.append({"Ciudad": city, "lat": lat, "lon": lon,
                         **{var: merged[var].corr(merged[TARGET]) for var in WEATHER_LABELS}})
        return pd.DataFrame(rows)

    st.markdown(
        "**Variables exógenas: clima de las 8 capitales de división.** Las variables de clima del modelo son el promedio "
        "simple, hora por hora, de las 8 capitales (Open-Meteo). El mapa muestra la correlación de cada ciudad con la "
        "demanda nacional en Train."
    )
    city_corr = load_city_correlations(train["datetime"], train[TARGET])
    map_var = st.segmented_control("Variable", list(WEATHER_LABELS), default="apparent_temperature",
                                   format_func=WEATHER_LABELS.get, key="map_var") or "apparent_temperature"
    national_r = float(train[map_var].corr(train[TARGET]))
    r_min, r_max = city_corr[map_var].min(), city_corr[map_var].max()
    accent = tuple(int(tokens["accent"][i:i + 2], 16) for i in (1, 3, 5))
    soft = tuple(int(tokens["mark"][i:i + 2], 16) for i in (1, 3, 5))
    span = (r_max - r_min) or 1
    city_map = city_corr.assign(
        r=city_corr[map_var].round(3),
        color=[[int(s + (a - s) * (r - r_min) / span) for s, a in zip(soft, accent)] + [230] for r in city_corr[map_var]],
        radius=12000 + 30000 * (city_corr[map_var].abs() / max(city_corr[map_var].abs().max(), 1e-9)),
    )[["Ciudad", "lat", "lon", "r", "color", "radius"]]
    st.pydeck_chart(
        pdk.Deck(
            layers=[
                pdk.Layer("ScatterplotLayer", city_map, get_position=["lon", "lat"], get_fill_color="color",
                          get_radius="radius", get_line_color=[40, 30, 60, 200], line_width_min_pixels=1, stroked=True,
                          pickable=True),
                pdk.Layer("TextLayer", city_map, get_position=["lon", "lat"], get_text="Ciudad", get_size=13,
                          get_color=[110, 100, 130, 255] if tokens is LIGHT else [220, 215, 235, 255],
                          get_pixel_offset=[0, -22]),
            ],
            initial_view_state=pdk.ViewState(latitude=23.8, longitude=90.3, zoom=5.6),
            tooltip={"text": "{Ciudad}\nr = {r}"},
            map_style=None,
        ),
        height=420,
    )
    caption(f"Correlación de {WEATHER_LABELS[map_var].lower()} con la demanda (Train) en cada capital: de "
            f"{r_min:.3f} a {r_max:.3f}. El promedio nacional que usa el modelo tiene r = {national_r:.3f}. "
            "El color y el tamaño del círculo indican la magnitud de la correlación.")

    heat = city_corr.melt(id_vars=["Ciudad", "lat", "lon"], var_name="var", value_name="r")
    heat = pd.concat([heat, pd.DataFrame({"Ciudad": "Promedio nacional", "var": list(WEATHER_LABELS),
                                          "r": [train[v].corr(train[TARGET]) for v in WEATHER_LABELS]})])
    heat["Variable"] = heat["var"].map(WEATHER_LABELS)
    city_order = ["Promedio nacional", *city_corr.sort_values(map_var, ascending=False)["Ciudad"]]
    heat_base = alt.Chart(heat).encode(
        x=alt.X("Variable:N", sort=list(WEATHER_LABELS.values()), title=None,
                axis=alt.Axis(labelAngle=0, orient="top", labelOverlap=False, labelExpr="split(datum.label, ' ')")),
        y=alt.Y("Ciudad:N", sort=city_order, title=None, axis=alt.Axis(labelLimit=200)),
    )
    st.altair_chart(
        (heat_base.mark_rect().encode(
            color=alt.Color("r:Q", scale=alt.Scale(scheme="purpleorange", domain=[-0.8, 0.8], reverse=True),
                            legend=alt.Legend(title="r")),
            tooltip=["Ciudad", "Variable", alt.Tooltip("r:Q", format=".3f")],
        ) + heat_base.mark_text(fontSize=11).encode(
            text=alt.Text("r:Q", format=".2f"),
            color=alt.condition(alt.datum.r > 0.5, alt.value("white"), alt.value("black")),
        )).properties(height=300),
        width="stretch",
    )
    best = city_corr.loc[city_corr["apparent_temperature"].idxmax()]
    caption("Correlación de cada variable de clima con la demanda en Train, por ciudad y para el promedio nacional. "
            f"La sensación térmica es la más correlacionada (máximo en {best['Ciudad']}, r = "
            f"{best['apparent_temperature']:.3f}). El promedio nacional queda dentro del rango de las ciudades: promediar "
            "no pierde señal y evita depender del clima de una sola ciudad.")
