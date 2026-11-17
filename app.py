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
