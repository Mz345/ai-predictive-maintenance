import os
import traceback

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Predictive Maintenance · AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / GLOBAL STYLES
# ============================================================
st.markdown(
    """
    <style>
    /* ---------- Base ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(59,130,246,0.10), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(168,85,247,0.10), transparent 55%),
            #0b1220;
    }
    .main .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #e5e7eb; }
    hr { border-color: rgba(148,163,184,0.15); }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0b1220 0%, #111a33 40%, #1b1147 100%);
        border: 1px solid rgba(148,163,184,0.18);
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.45);
    }
    .hero::before {
        content: "";
        position: absolute;
        inset: -40% -20% auto auto;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(139,92,246,0.35), transparent 60%);
        filter: blur(20px);
        pointer-events: none;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #f8fafc;
    }
    .hero p {
        margin: 0.6rem 0 1rem 0;
        max-width: 720px;
        font-size: 0.98rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-right: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #e2e8f0;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
    }
    .pill.green { color:#6ee7b7; border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.10); }
    .pill.blue  { color:#93c5fd; border-color: rgba(59,130,246,0.35); background: rgba(59,130,246,0.10); }
    .pill.violet{ color:#c4b5fd; border-color: rgba(139,92,246,0.35); background: rgba(139,92,246,0.10); }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 0.78rem;
        letter-spacing: 1.4px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: linear-gradient(180deg, rgba(30,41,59,0.85), rgba(15,23,42,0.85));
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
        box-shadow: 0 6px 20px rgba(0,0,0,0.28);
    }
    .metric-card .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        font-weight: 700;
        color: #94a3b8;
    }
    .metric-card .value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.15;
        margin-top: 0.2rem;
    }
    .metric-card .sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* ---------- Result cards ---------- */
    .result {
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        margin-top: 0.5rem;
        border: 1px solid transparent;
        box-shadow: 0 14px 40px rgba(0,0,0,0.35);
    }
    .result h2 { margin: 0; font-size: 1.5rem; font-weight: 800; color: #ffffff; }
    .result p  { margin-top: 0.5rem; color: #f1f5f9; opacity: 0.92; line-height: 1.55; font-size: 0.95rem; }
    .result.danger {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 50%, #b91c1c 100%);
        border-color: rgba(239,68,68,0.45);
    }
    .result.safe {
        background: linear-gradient(135deg, #052e2b 0%, #064e3b 50%, #047857 100%);
        border-color: rgba(16,185,129,0.45);
    }
    .result .risk {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        background: rgba(255,255,255,0.12);
        color: #ffffff;
        letter-spacing: 0.5px;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: rgba(15,23,42,0.9) !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: rgba(139,92,246,0.55) !important;
    }
    div[data-baseweb="select"] svg { fill: #94a3b8 !important; }

    /* ---------- Button ---------- */
    .stButton > button, .stFormSubmitButton > button {
        width: 100%;
        padding: 0.9rem 1rem;
        font-weight: 700;
        font-size: 1rem;
        color: #ffffff !important;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        border: none;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(124,58,237,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(124,58,237,0.5);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: rgba(11,18,32,0.97);
        border-right: 1px solid rgba(148,163,184,0.14);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

    /* ---------- Progress ---------- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22d3ee 0%, #2563eb 50%, #7c3aed 100%);
    }

    /* ---------- Expander ---------- */
    details {
        border-radius: 12px !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        background: rgba(15,23,42,0.5) !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        line-height: 1.7;
        margin-top: 2.5rem;
    }
    .footer strong { color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_PATH = "best_model.pkl"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

MODEL_METRICS = {
    "Accuracy": 0.9855,
    "Precision": 0.8824,
    "Recall": 0.6618,
    "F1-Score": 0.7563,
}


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def load_model_or_die():
    """Loads the model once and displays a friendly error otherwise."""
    if "model" in st.session_state:
        return st.session_state.model

    if not os.path.exists(MODEL_PATH):
        st.markdown(
            """
            <div class="hero">
                <h1>⚠️ Model file not found</h1>
                <p>
                    The file <code>best_model.pkl</code> was not found in the app
                    directory. Please include it in the repository before deploying.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error("❌ Unable to load the trained model.")
        st.code(str(exc))
        st.info(
            "Make sure `best_model.pkl` exists and that the scikit-learn version "
            "used at training time matches the one installed in this environment."
        )
        st.stop()

    st.session_state.model = model
    return model


model = load_model_or_die()


# ============================================================
# HELPERS
# ============================================================
def predict(input_df: pd.DataFrame):
    """Return (prediction, failure_probability)."""
    prediction = int(model.predict(input_df)[0])

    failure_prob = None
    if hasattr(model, "predict_proba"):
        try:
            failure_prob = float(model.predict_proba(input_df)[0][1])
        except Exception:
            failure_prob = None

    return prediction, failure_prob


def risk_level(prob: float | None) -> str:
    if prob is None:
        return "Unknown"
    if prob >= 0.66:
        return "High"
    if prob >= 0.33:
        return "Moderate"
    return "Low"


def render_metric(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {f'<div class="sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>⚙️ AI-Based Predictive Maintenance</h1>
        <p>
            Estimate whether an industrial machine is likely to fail from its live
            operating conditions. Enter the sensor readings below and the trained
            <strong>Gradient Boosting</strong> pipeline will compute the failure risk.
        </p>
        <span class="pill green">● Model ready</span>
        <span class="pill blue">Accuracy 98.55%</span>
        <span class="pill violet">F1-Score 0.756</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📘 About this app")
    st.caption(
        "A machine-learning pipeline trained on the AI4I 2020 Predictive Maintenance "
        "dataset. It classifies machine status as **Healthy** or **Failure** based on "
        "six operating features."
    )

    st.divider()
    st.markdown("### 🎯 Model performance")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{MODEL_METRICS['Accuracy']*100:.2f}%")
    c1.metric("Precision", f"{MODEL_METRICS['Precision']*100:.2f}%")
    c2.metric("Recall", f"{MODEL_METRICS['Recall']*100:.2f}%")
    c2.metric("F1-Score", f"{MODEL_METRICS['F1-Score']:.3f}")

    st.divider()
    st.markdown("### 🧪 Input features")
    st.markdown(
        """
        | Feature | Unit |
        |---|---|
        | Type | L / M / H |
        | Air temperature | Kelvin |
        | Process temperature | Kelvin |
        | Rotational speed | rpm |
        | Torque | Nm |
        | Tool wear | min |
        """
    )

    st.divider()
    st.caption("Stack: Streamlit · scikit-learn · pandas")

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("## 🧾 Machine Parameters")

with st.form("input_form", clear_on_submit=False):

    # ---- Row 1: Machine identity + temperatures ----
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section-label">Machine</div>', unsafe_allow_html=True)
        machine_type = st.selectbox(
            "Type",
            options=["L", "M", "H"],
            index=0,
            help="Product quality tier: L = Low, M = Medium, H = High.",
        )

        st.markdown('<div class="section-label" style="margin-top:1rem;">Temperature</div>',
                    unsafe_allow_html=True)
        air_temp = st.number_input(
            "Air temperature (K)",
            min_value=250.0, max_value=350.0, value=300.0, step=0.1,
            help="Ambient air temperature around the machine.",
        )
        process_temp = st.number_input(
            "Process temperature (K)",
            min_value=250.0, max_value=400.0, value=310.0, step=0.1,
            help="Temperature of the manufacturing process.",
        )

    with right:
        st.markdown('<div class="section-label">Mechanics</div>', unsafe_allow_html=True)
        rotational_speed = st.number_input(
            "Rotational speed (rpm)",
            min_value=0, max_value=10000, value=1500, step=10,
            help="Rotational speed of the machine spindle.",
        )
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0, max_value=200.0, value=40.0, step=0.1,
            help="Rotational force applied to the tool.",
        )
        tool_wear = st.number_input(
            "Tool wear (min)",
            min_value=0, max_value=500, value=100, step=1,
            help="Cumulative usage time of the cutting tool.",
        )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Machine Failure")

# ============================================================
# PREDICTION OUTPUT
# ============================================================
if submitted:
    input_df = pd.DataFrame(
        [{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction, failure_prob = predict(input_df)

        st.markdown("## 📊 Prediction")

        # ---------- Result banner ----------
        if prediction == 1:
            risk = risk_level(failure_prob)
            prob_txt = f"Estimated failure probability: <strong>{failure_prob*100:.2f}%</strong>." \
                if failure_prob is not None else ""
            st.markdown(
                f"""
                <div class="result danger">
                    <h2>⚠️ Machine Failure Predicted</h2>
                    <p>
                        The model detected operating conditions consistent with a
                        potential failure. Inspect the machine and schedule preventive
                        maintenance before continuing operation.
                        {prob_txt}
                    </p>
                    <span class="risk">Risk level: {risk}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            risk = risk_level(failure_prob)
            prob_txt = f"Estimated failure probability: <strong>{failure_prob*100:.2f}%</strong>." \
                if failure_prob is not None else ""
            st.markdown(
                f"""
                <div class="result safe">
                    <h2>✅ No Machine Failure Predicted</h2>
                    <p>
                        The machine appears to be running under healthy conditions.
                        Continue normal operation and keep monitoring sensor readings.
                        {prob_txt}
                    </p>
                    <span class="risk">Risk level: {risk}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- KPI cards ----------
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)

        with k1:
            render_metric(
                "Predicted Class",
                "Failure" if prediction == 1 else "Healthy",
                "Binary output from the model",
            )
        with k2:
            render_metric(
                "Failure Probability",
                f"{failure_prob*100:.2f}%" if failure_prob is not None else "N/A",
                "Model confidence estimate",
            )
        with k3:
            render_metric("Risk Level", risk, "Derived from probability")

        # ---------- Probability bar ----------
        if failure_prob is not None:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Failure probability</div>',
                        unsafe_allow_html=True)
            st.progress(min(max(failure_prob, 0.0), 1.0))
            st.caption(
                f"The model estimates a **{failure_prob*100:.2f}%** probability of "
                f"failure for the provided operating conditions."
            )

        # ---------- Input summary ----------
        with st.expander("📋 View entered parameters"):
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

    except Exception as exc:
        st.error("❌ An error occurred while making the prediction.")
        st.code(str(exc))
        st.caption(
            "Check that the input column names match the training features "
            "and that the preprocessing pipeline expects the same schema."
        )
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc())

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        <strong>AI-Based Predictive Maintenance</strong><br>
        Gradient Boosting · scikit-learn · Streamlit<br>
        <em>For informational purposes only — always confirm with physical inspection.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
