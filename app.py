import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import sys
import types
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt

try:
    import hopsworks
except Exception:  # pragma: no cover
    hopsworks = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from pathlib import Path

import numpy.random._pickle as _npr_pickle
from numpy.random import MT19937

# Fix: numpy version mismatch causes BitGenerator class object (instead of its name)
# to be passed during unpickling of models saved with numpy>=2.0
_original_ctor = _npr_pickle.__bit_generator_ctor

def _patched_bit_generator_ctor(bit_generator_name=MT19937):
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    return _original_ctor(bit_generator_name)

_npr_pickle.__bit_generator_ctor = _patched_bit_generator_ctor
def load_env_file(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def normalize_secret(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        for key in ("api_key", "value", "key"):
            if key in value and value[key] not in (None, ""):
                value = value[key]
                break
        else:
            return None
    value = str(value).strip()
    return value if value else None


def find_nested_secret(data, key_names):
    if isinstance(data, dict):
        for key in list(data.keys()):
            lowered = str(key).lower()
            if lowered in {name.lower() for name in key_names}:
                return normalize_secret(data[key])
            if isinstance(data[key], dict):
                nested = find_nested_secret(data[key], key_names)
                if nested:
                    return nested
            if isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        nested = find_nested_secret(item, key_names)
                        if nested:
                            return nested
    return None


def load_local_secret_file_candidates():
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ]
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        yield path


def load_secret_from_files():
    for candidate in load_local_secret_file_candidates():
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".toml":
                try:
                    import tomllib
                except ModuleNotFoundError:
                    tomllib = None
                if tomllib is not None:
                    with candidate.open("rb") as fh:
                        data = tomllib.load(fh)
                    secret = find_nested_secret(data, ["HOPSWORKS_API_KEY", "hopsworks_api_key", "hopsworks"])
                    if secret:
                        return secret
            else:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key.lower() == "hopsworks_api_key" and value:
                        return value
        except Exception:
            pass
    return None


for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
    load_env_file(candidate)

secret_from_files = load_secret_from_files()
if secret_from_files and not os.environ.get("HOPSWORKS_API_KEY"):
    os.environ["HOPSWORKS_API_KEY"] = secret_from_files

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass

st.set_page_config(
    page_title="10Pearl AQI Predictors — Sargodha",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Force dark background regardless of Streamlit's own theme / DOM version */
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .stApp {
        background: radial-gradient(circle at top left, #101827 0%, #05070d 60%) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    body, p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #e6edf3;
    }

    /* Hero header */
    .hero-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 28px 32px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
    }
    .brand-tag {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #7fd8ff !important;
        background: rgba(127,216,255,0.12);
        border: 1px solid rgba(127,216,255,0.35);
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .hero-sub {
        font-size: 15px;
        color: #c9d6df !important;
        margin-top: 6px;
    }

    /* Metric cards — solid background so text is always visible, independent of theme */
    .metric-card {
        background: #131c2e;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255,255,255,0.3);
    }
    .metric-label {
        font-size: 13px;
        color: #9fb0c3 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff !important;
    }

    /* Big AQI display */
    .aqi-hero {
        border-radius: 24px;
        padding: 34px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.12);
        background-color: #10192b;
    }
    .aqi-hero-number {
        font-size: 96px;
        font-weight: 800;
        line-height: 1;
        text-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .aqi-hero-label {
        font-size: 22px;
        font-weight: 600;
        margin-top: 6px;
        letter-spacing: 0.5px;
    }
    .aqi-hero-date {
        font-size: 14px;
        color: rgba(255,255,255,0.85) !important;
        margin-top: 4px;
    }

    .day-tag {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        color: #fff !important;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .hazard-banner {
        padding: 20px 26px;
        border-radius: 16px;
        font-size: 17px;
        font-weight: 600;
        color: white !important;
        margin: 18px 0;
        animation: pulseGlow 1.8s infinite;
        border: 1px solid rgba(255,255,255,0.2);
    }
    @keyframes pulseGlow {
        0%   { box-shadow: 0 0 0px rgba(255,0,0,0.4); }
        50%  { box-shadow: 0 0 28px rgba(255,0,0,0.55); }
        100% { box-shadow: 0 0 0px rgba(255,0,0,0.4); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #05070d 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 18px;
        border: none;
        background: linear-gradient(135deg, #2c5364, #0f2027);
        color: white !important;
        transition: 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(44,83,100,0.5);
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #2c5364; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def create_requests_session(retries=3, backoff_factor=1.0):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Compatibility shim for old scikit-learn pickles that reference the legacy `_loss` module.
def ensure_sklearn_loss_compat():
    try:
        import sklearn._loss._loss as sklearn_loss
        if "_loss" not in sys.modules:
            _loss_module = types.ModuleType("_loss")
            for attr in dir(sklearn_loss):
                if not attr.startswith("_"):
                    setattr(_loss_module, attr, getattr(sklearn_loss, attr))
            sys.modules["_loss"] = _loss_module
    except Exception:
        pass

ensure_sklearn_loss_compat()

# ============================================================
# AQI CATEGORY HELPERS (US AQI breakpoints)
# ============================================================
def get_aqi_category(aqi):
    """Return (label, color_hex, emoji, advisory) for a given US AQI value."""
    if aqi <= 50:
        return "Good", "#2ecc71", "🟢", "Air quality is satisfactory. Enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", "#f1c40f", "🟡", "Acceptable, but sensitive groups should watch symptoms."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22", "🟠", "Children, elderly & respiratory patients should limit outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "#e74c3c", "🔴", "Everyone may experience health effects. Reduce prolonged outdoor exertion."
    elif aqi <= 300:
        return "Very Unhealthy", "#8e44ad", "🟣", "Health alert! Avoid outdoor activity, wear a mask if going outside."
    else:
        return "Hazardous", "#7d1414", "🟤", "Emergency conditions! Stay indoors, use air purifiers, avoid all outdoor exposure."


def aqi_gauge_chart(value, title):
    label, color, emoji, _ = get_aqi_category(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 40, 'color': color}},
        title={'text': f"{emoji} {title}", 'font': {'size': 16, 'color': '#dfe9f3'}},
        gauge={
            'axis': {'range': [0, 500], 'tickcolor': '#8fa3b3', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(46,204,113,0.25)'},
                {'range': [50, 100], 'color': 'rgba(241,196,15,0.25)'},
                {'range': [100, 150], 'color': 'rgba(230,126,34,0.25)'},
                {'range': [150, 200], 'color': 'rgba(231,76,60,0.25)'},
                {'range': [200, 300], 'color': 'rgba(142,68,173,0.25)'},
                {'range': [300, 500], 'color': 'rgba(125,20,20,0.25)'},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#dfe9f3"},
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


# ============================================================
# HERO HEADER
# ============================================================
st.markdown(f"""
<div class="hero-box">
    <p class="brand-tag">10PEARL AQI PREDICTORS</p>
    <p class="hero-title">🌬️ Sargodha AQI — 3-Day Live Forecast System</p>
    <p class="hero-sub">📍 Sargodha, Punjab, Pakistan &nbsp;|&nbsp; 🗓️ {datetime.now().strftime('%A, %d %B %Y')} &nbsp;|&nbsp; Powered by Hopsworks + Open-Meteo</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HOPSWORKS API KEY — read from env / .env / Streamlit secrets
# and allow manual entry in the sidebar for deployment use.
# ============================================================
def resolve_hopsworks_api_key():
    env_key = normalize_secret(os.environ.get("HOPSWORKS_API_KEY"))
    if env_key:
        return env_key

    local_key = load_secret_from_files()
    if local_key:
        os.environ["HOPSWORKS_API_KEY"] = local_key
        return local_key

    try:
        secret_key = normalize_secret(st.secrets.get("HOPSWORKS_API_KEY"))
        if secret_key:
            return secret_key
        secret_key = normalize_secret(st.secrets.get("hopsworks_api_key"))
        if secret_key:
            return secret_key
        if "hopsworks" in st.secrets:
            nested = normalize_secret(st.secrets["hopsworks"].get("api_key"))
            if nested:
                return nested
    except Exception:
        pass

    sidebar_key = st.sidebar.text_input(
        "Hopsworks API key",
        type="password",
        help="Paste Hopsworks key here if it is not loaded from the environment, local secret files, or Streamlit secrets.",
        key="hopsworks_api_key_input",
    )
    if sidebar_key and str(sidebar_key).strip():
        return str(sidebar_key).strip()

    return None


st.sidebar.markdown("## 🔑 Hopsworks Connection")
api_key = resolve_hopsworks_api_key()
if api_key:
    st.sidebar.success("✅ Hopsworks API key loaded.")
else:
    st.sidebar.warning("⚠️ HOPSWORKS_API_KEY not found. Set it in .env, Streamlit secrets, or paste it here.")
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This dashboard pulls live weather & air-quality data from Open-Meteo, "
    "assembles model features, and runs 3 pre-trained Gradient Boosting "
    "models (Day 1 / Day 2 / Day 3) hosted on Hopsworks to forecast AQI."
)

# Sargodha Coordinates
LATITUDE = 32.0836
LONGITUDE = 72.6711

# Helper function to extract predict-capable model object
def extract_model(obj):
    if hasattr(obj, "predict"):
        return obj
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        for item in obj:
            if hasattr(item, "predict"):
                return item
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if hasattr(val, "predict"):
                return val
    return None


def is_legacy_numpy_state_error(exc):
    if exc is None:
        return False
    msg = str(exc).lower()
    legacy_markers = (
        "state is not a legacy",
        "legacy mt19937",
        "bitgenerator",
        "bit_generator",
        "state must be for a mt19937 prng",
    )
    return any(marker in msg for marker in legacy_markers)


def load_compat_model_file(file_path):
    try:
        return joblib.load(file_path)
    except Exception as exc:
        if is_legacy_numpy_state_error(exc):
            raise RuntimeError(
                "This model artifact was pickled with an older NumPy random-state format and is incompatible with the current environment. "
                "Re-train or re-upload the model using the project’s pinned NumPy/scikit-learn stack."
            ) from exc
        raise


def compute_shap_contributions(model, feature_names, x_vec):
    if shap is None:
        return pd.DataFrame({
            "feature": feature_names,
            "shap_value": np.zeros(len(feature_names), dtype=float),
            "abs_shap": np.zeros(len(feature_names), dtype=float),
        })

    try:
        explainer = shap.Explainer(model)
        explanation = explainer(x_vec)
        if isinstance(explanation, list):
            explanation = explanation[0]

        values = explanation.values
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]

        names = list(getattr(explanation, "feature_names", feature_names) or feature_names)
        contrib_df = pd.DataFrame({
            "feature": names,
            "shap_value": np.asarray(values, dtype=float),
        })
        contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
        return contrib_df.sort_values("abs_shap", ascending=False)
    except Exception:
        return pd.DataFrame({
            "feature": feature_names,
            "shap_value": np.zeros(len(feature_names), dtype=float),
            "abs_shap": np.zeros(len(feature_names), dtype=float),
        })


def render_shap_feature_chart(contrib_df, title):
    if shap is None:
        st.caption(f"SHAP is not installed; explainability is unavailable for {title}.")
        return

    top = contrib_df.head(10).copy()
    if top.empty:
        st.caption(f"SHAP data unavailable for {title}.")
        return

    top = top.sort_values("shap_value", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["shap_value"], color=["#5dade2" if v >= 0 else "#f39c12" for v in top["shap_value"]])
    ax.axvline(0, color="#dfe9f3", linewidth=1)
    ax.set_title(f"Top SHAP Drivers — {title}", color="#e6edf3", fontsize=12)
    ax.set_xlabel("SHAP value", color="#dfe9f3")
    ax.set_ylabel("Feature", color="#dfe9f3")
    ax.tick_params(axis="x", colors="#dfe9f3")
    ax.tick_params(axis="y", colors="#dfe9f3")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_color("#2c5364")
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#10192b")
    fig.tight_layout()
    st.pyplot(fig)

# --- 0. BUILD FEATURE VECTOR FOR EACH MODEL HORIZON ---
@st.cache_data(ttl=86400)
def fetch_historical_daily_data(lat, lon, lookback_days=45):
    end_date = datetime.utcnow().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)

    session = create_requests_session(retries=4, backoff_factor=1.0)
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto",
    }
    w_res = session.get(weather_url, params=weather_params, timeout=60)
    w_res.raise_for_status()
    weather_json = w_res.json()
    if "hourly" not in weather_json:
        raise ValueError(f"Weather history not available: {weather_json}")

    weather_df = pd.DataFrame(weather_json["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.set_index("time").resample("D").mean()

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": "auto",
    }
    aqi_res = session.get(aqi_url, params=aqi_params, timeout=60)
    aqi_res.raise_for_status()
    aqi_json = aqi_res.json()
    if "hourly" not in aqi_json:
        raise ValueError(f"AQI history not available: {aqi_json}")

    aqi_df = pd.DataFrame(aqi_json["hourly"])
    aqi_df["time"] = pd.to_datetime(aqi_df["time"])
    aqi_df = aqi_df.set_index("time").resample("D").mean()
    aqi_df = aqi_df.rename(columns={"us_aqi": "AQI"})

    combined = pd.concat([weather_df, aqi_df[["AQI", "pm2_5"]]], axis=1)
    combined = combined[["AQI", "pm2_5", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]]
    combined = combined.dropna()
    if len(combined) < 35:
        raise ValueError("Not enough historical daily data available for lag/rolling feature construction.")
    return combined

@st.cache_data(ttl=600)
def fetch_forecast_weather(lat, lon, horizon_days=5):
    session = create_requests_session(retries=4, backoff_factor=1.0)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": horizon_days,
        "timezone": "auto",
    }
    forecast_res = session.get(forecast_url, params=forecast_params, timeout=60)
    forecast_res.raise_for_status()
    forecast_json = forecast_res.json()
    if "hourly" not in forecast_json:
        raise ValueError(f"Weather forecast not available: {forecast_json}")

    fc_df = pd.DataFrame(forecast_json["hourly"])
    fc_df["time"] = pd.to_datetime(fc_df["time"])
    fc_df = fc_df.set_index("time").resample("D").mean()
    return fc_df


def build_feature_vector(historical, forecast_daily, feature_names):
    latest_date = historical.index.max().date()
    latest_row = historical.loc[historical.index.date == latest_date]
    if latest_row.empty:
        raise ValueError("Latest historical date missing from assembled daily data.")
    latest_row = latest_row.iloc[0]

    def get_lag(name, lag):
        date = latest_date - timedelta(days=lag)
        if date not in historical.index.date:
            raise ValueError(f"Missing historical data for lag {lag} days ago ({date}).")
        return historical.loc[historical.index.date == date].iloc[0][name]

    def get_roll(stat, window):
        series = historical["AQI"].shift(1).rolling(window=window)
        value = getattr(series, stat)().iloc[-1]
        if pd.isna(value):
            raise ValueError(f"Missing rolling feature {stat} over {window} days.")
        return value

    feature_vector = []
    for feature in feature_names:
        if match := re.match(r"aqi_lag_(\d+)$", feature):
            feature_vector.append(get_lag("AQI", int(match.group(1))))
        elif match := re.match(r"pm25_lag_(\d+)$", feature):
            feature_vector.append(get_lag("pm2_5", int(match.group(1))))
        elif match := re.match(r"temp_lag_(\d+)$", feature):
            feature_vector.append(get_lag("temperature_2m", int(match.group(1))))
        elif match := re.match(r"aqi_roll_(mean|max|min|std)_(\d+)$", feature):
            stat = match.group(1)
            window = int(match.group(2))
            feature_vector.append(get_roll(stat, window))
        elif feature == "sin_day":
            day_of_year = latest_date.timetuple().tm_yday
            feature_vector.append(np.sin(2 * np.pi * day_of_year / 365.25))
        elif feature == "cos_day":
            day_of_year = latest_date.timetuple().tm_yday
            feature_vector.append(np.cos(2 * np.pi * day_of_year / 365.25))
        elif feature == "month":
            feature_vector.append(latest_date.month)
        elif feature == "dayofweek":
            feature_vector.append(latest_date.weekday())
        elif match := re.match(r"(temp|humidity|wind)_future_h(\d+)$", feature):
            kind = match.group(1)
            horiz = int(match.group(2))
            target_date = latest_date + timedelta(days=horiz)
            if target_date not in forecast_daily.index.date:
                raise ValueError(f"Forecast weather not available for {target_date}.")
            weather_row = forecast_daily.loc[forecast_daily.index.date == target_date].iloc[0]
            if kind == "temp":
                feature_vector.append(weather_row["temperature_2m"])
            elif kind == "humidity":
                feature_vector.append(weather_row["relative_humidity_2m"])
            else:
                feature_vector.append(weather_row["wind_speed_10m"])
        else:
            raise ValueError(f"Unknown feature name: {feature}")

    return np.array(feature_vector).reshape(1, -1)

# --- 1. OPEN-METEO API FETCHING WITH BETTER TIMEOUT & RETRY ---
@st.cache_data(ttl=600)
def fetch_live_weather_and_aqi():
    data = {"pm25": 45.0, "pm10": 90.0, "temp": 30.0, "humidity": 55.0, "wind": 10.0}

    # Weather Fetch
    session = create_requests_session(retries=3, backoff_factor=0.5)
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        }
        w_res = session.get(weather_url, params=weather_params, timeout=30)
        w_res.raise_for_status()
        w_json = w_res.json()
        curr_w = w_json.get("current", {})
        data["temp"] = curr_w.get("temperature_2m", 30.0)
        data["humidity"] = curr_w.get("relative_humidity_2m", 55.0)
        data["wind"] = curr_w.get("wind_speed_10m", 10.0)
    except Exception as e:
        st.warning(f"Weather API Warning: {e}. Fallback values used.")

    # Air Quality Fetch
    try:
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality"
        aqi_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "pm10,pm2_5",
        }
        a_res = session.get(aqi_url, params=aqi_params, timeout=30)
        a_res.raise_for_status()
        a_json = a_res.json()
        curr_a = a_json.get("current", {})
        data["pm25"] = curr_a.get("pm2_5", 45.0)
        data["pm10"] = curr_a.get("pm10", 90.0)
    except Exception as e:
        st.warning(f"Air Quality API Warning: {e}. Fallback values used.")

    return data

# --- 2. HOPSWORKS MODEL LOADING WITH SAFELY UNPACKED MODEL OBJECT ---
@st.cache_resource(show_spinner="Hopsworks Registry se Models download aur extract ho rahe hain...")
def load_all_models(key):
    if hopsworks is None:
        return {}

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=key,
        project="colab"
    )
    mr = project.get_model_registry()

    models_dict = {}
    model_info = [
        ("sargodha_aqi_gbr_day1", 4, "Day 1"),
        ("sargodha_aqi_gbr_day2", 3, "Day 2"),
        ("sargodha_aqi_gbr_day3", 3, "Day 3")
    ]

    for m_name, m_ver, label in model_info:
        try:
            model_meta = mr.get_model(m_name, version=m_ver)
            model_dir = model_meta.download()

            model_obj = None
            feature_names = None
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file == "features.pkl":
                        feature_names = load_compat_model_file(file_path)
                    elif file.endswith(".pkl") or file.endswith(".joblib"):
                        if file == "features.pkl":
                            continue
                        try:
                            raw_obj = load_compat_model_file(file_path)
                        except RuntimeError as exc:
                            raise ValueError(
                                f"{m_name} model artifact is incompatible with the current NumPy/scikit-learn stack: {exc}"
                            ) from exc
                        extracted = extract_model(raw_obj)
                        if extracted is not None:
                            model_obj = extracted
                if model_obj is not None and feature_names is not None:
                    break

            if model_obj is None:
                raise ValueError(f"No model object found in downloaded artifact for {m_name}.")
            if feature_names is None:
                raise ValueError(f"No features.pkl found for {m_name}; cannot build the model input vector.")

            models_dict[label] = {
                "model": model_obj,
                "features": feature_names,
            }
        except Exception as exc:
            st.sidebar.warning(
                f"⚠️ {m_name} model could not be loaded because the Hopsworks artifact is incompatible with the current NumPy/scikit-learn environment. "
                f"Re-train or re-upload the model from the same stack. Details: {exc}"
            )
            continue

    return models_dict

models = None
if api_key:
    if hopsworks is None:
        st.sidebar.warning(
            "⚠️ Hopsworks library is not installed in this deployment. The app will run without forecast models until Hopsworks is available."
        )
        models = {}
    else:
        try:
            models = load_all_models(api_key)
            if models:
                st.sidebar.success("✅ Day 1, 2, 3 Models Ready!")
            else:
                st.sidebar.warning("⚠️ No forecast models loaded from Hopsworks.")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Forecast models could not be loaded: {e}")
            import traceback
            st.sidebar.code(traceback.format_exc())
            models = {}
else:
    st.info("👈 HOPSWORKS_API_KEY set nahi hai. Sidebar me API key paste karo ya .env file me save karo.")

# --- 3. LIVE DATA UI ---
st.markdown("### 📡 Live Weather & Air Quality — Sargodha")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()

live_data = fetch_live_weather_and_aqi()

m1, m2, m3, m4, m5 = st.columns(5)
metric_defs = [
    (m1, "PM2.5", f"{live_data['pm25']:.1f} µg/m³", "🫧"),
    (m2, "PM10", f"{live_data['pm10']:.1f} µg/m³", "🌫️"),
    (m3, "Temperature", f"{live_data['temp']:.1f} °C", "🌡️"),
    (m4, "Humidity", f"{live_data['humidity']:.0f} %", "💧"),
    (m5, "Wind Speed", f"{live_data['wind']:.1f} km/h", "🍃"),
]
for col, label, value, icon in metric_defs:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- 4. PREDICTION & ALARMING ALERT SYSTEM ---
st.markdown("### 🚀 Run Forecast")
run_forecast = st.button("🚀 Run 3-Day AQI Forecast", type="primary")

if run_forecast:
    if not models or len(models) < 3:
        st.error("Models load nahi hue! HOPSWORKS_API_KEY environment variable verify karo.")
    else:
        try:
            with st.spinner("Fetching historical & forecast weather data, building features, running models..."):
                historical = fetch_historical_daily_data(LATITUDE, LONGITUDE)
                forecast_daily = fetch_forecast_weather(LATITUDE, LONGITUDE, horizon_days=5)

                preds = []
                for label in ["Day 1", "Day 2", "Day 3"]:
                    model_info = models[label]
                    model = model_info["model"]
                    feature_names = model_info["features"]
                    x_vec = build_feature_vector(historical, forecast_daily, feature_names)
                    preds.append(float(model.predict(x_vec)[0]))

            today = datetime.now()
            dates = [(today + timedelta(days=i + 1)) for i in range(3)]
            date_labels = [d.strftime("%d %b (%a)") for d in dates]
            preds = [round(v, 1) for v in preds]

            # ---------------- HAZARD BANNER ----------------
            max_aqi = max(preds)
            worst_day_idx = preds.index(max_aqi)
            worst_label, worst_color, worst_emoji, worst_advisory = get_aqi_category(max_aqi)

            if max_aqi > 200:
                st.markdown(f"""
                <div class="hazard-banner" style="background: linear-gradient(135deg, #7d1414, #b71c1c);">
                    🚨 SEVERE ALERT — {worst_label.upper()} AIR EXPECTED ON {date_labels[worst_day_idx]}!<br>
                    Peak Predicted AQI: <b>{max_aqi}</b> {worst_emoji} &nbsp;|&nbsp; {worst_advisory}
                </div>
                """, unsafe_allow_html=True)
            elif max_aqi > 150:
                st.markdown(f"""
                <div class="hazard-banner" style="background: linear-gradient(135deg, #e67e22, #d35400);">
                    ⚠️ UNHEALTHY AIR QUALITY WARNING — {date_labels[worst_day_idx]}<br>
                    Peak Predicted AQI: <b>{max_aqi}</b> {worst_emoji} &nbsp;|&nbsp; {worst_advisory}
                </div>
                """, unsafe_allow_html=True)
            elif max_aqi > 100:
                st.markdown(f"""
                <div class="hazard-banner" style="background: linear-gradient(135deg, #f39c12, #f1c40f); color:#3a2c00;">
                    🟡 MODERATE ALERT — Sensitive groups take care on {date_labels[worst_day_idx]}<br>
                    Peak Predicted AQI: <b>{max_aqi}</b> {worst_emoji}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ **ACCEPTABLE AIR QUALITY.** Maximum predicted AQI over next 3 days is **{max_aqi}** {worst_emoji}")

            # ---------------- BIG AQI HERO CARDS ----------------
            st.markdown("### 📅 3-Day AQI Forecast")
            hcols = st.columns(3)
            for i, col in enumerate(hcols):
                label, color, emoji, advisory = get_aqi_category(preds[i])
                with col:
                    st.markdown(f"""
                    <div class="aqi-hero" style="background: linear-gradient(160deg, {color}33, {color}0d);">
                        <span class="day-tag">📆 Day {i+1} · {date_labels[i]}</span>
                        <div class="aqi-hero-number" style="color:{color};">{preds[i]:.0f}</div>
                        <div class="aqi-hero-label" style="color:{color};">{emoji} {label}</div>
                        <div class="aqi-hero-date">{advisory}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ---------------- SHAP EXPLAINABILITY ----------------
            st.markdown("### 🔍 SHAP Feature Impact")
            forecast_inputs = []
            for label in ["Day 1", "Day 2", "Day 3"]:
                model_info = models[label]
                x_vec = build_feature_vector(historical, forecast_daily, model_info["features"])
                forecast_inputs.append({
                    "label": label,
                    "model": model_info["model"],
                    "features": model_info["features"],
                    "x_vec": x_vec,
                })

            for item in forecast_inputs:
                contrib = compute_shap_contributions(item["model"], item["features"], item["x_vec"])
                st.subheader(f"{item['label']} — Top contributors")
                render_shap_feature_chart(contrib, item["label"])

            # ---------------- GAUGE CHARTS ----------------
            st.markdown("### 🧭 AQI Gauges")
            gcols = st.columns(3)
            for i, col in enumerate(gcols):
                with col:
                    st.plotly_chart(aqi_gauge_chart(preds[i], f"Day {i+1} — {date_labels[i]}"), use_container_width=True)

            # ---------------- TREND CHART: HISTORY + FORECAST ----------------
            st.markdown("### 📈 AQI Trend — Last 14 Days + 3-Day Model Forecast")

            hist_tail = historical.tail(14).copy()
            hist_dates = list(hist_tail.index.date)
            hist_values = list(hist_tail["AQI"].values)

            trend_fig = go.Figure()

            # Historical actual AQI
            trend_fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_values,
                mode="lines+markers",
                name="Historical AQI",
                line=dict(color="#5dade2", width=3),
                marker=dict(size=6),
            ))

            # Bridge point connecting history to forecast
            bridge_x = [hist_dates[-1]] + [d.date() for d in dates]
            bridge_y = [hist_values[-1]] + preds

            trend_fig.add_trace(go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines+markers",
                name="Model Forecast",
                line=dict(color="#f39c12", width=3, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
            ))

            # Hazard threshold reference lines
            for level, color, name in [(100, "#f1c40f", "Moderate"), (150, "#e67e22", "Unhealthy (Sensitive)"),
                                        (200, "#e74c3c", "Unhealthy"), (300, "#8e44ad", "Very Unhealthy")]:
                trend_fig.add_hline(y=level, line_dash="dot", line_color=color, opacity=0.5,
                                     annotation_text=name, annotation_font_color=color, annotation_font_size=10)

            trend_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#dfe9f3"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Date"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI"),
                height=420,
            )
            st.plotly_chart(trend_fig, use_container_width=True)

            # ---------------- RAW TABLE ----------------
            with st.expander("📋 View Raw Forecast Table"):
                df_res = pd.DataFrame({
                    "Forecast Date": date_labels,
                    "Predicted AQI": preds,
                    "Category": [get_aqi_category(p)[0] for p in preds],
                })
                st.dataframe(df_res, use_container_width=True, hide_index=True)

        except Exception as err:
            st.error(f"Prediction Error: {err}")