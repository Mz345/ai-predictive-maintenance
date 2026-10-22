import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #e2e8f0 !important;
    }

    /* ---------- Hero header ---------- */
    .hero {
        background: linear-gradient(120deg, #1e3a8a 0%, #4c1d95 50%, #831843 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.35);
        border: 1px solid rgba(148,163,184,0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    .hero p {
        margin-top: 0.6rem;
        font-size: 1.02rem;
        color: #cbd5e1 !important;
        max-width: 720px;
        line-height: 1.55;
    }
    .hero .badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 6px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        font-size: 0.78rem;
        font-weight: 600;
        color: #f8fafc !important;
        letter-spacing: 0.4px;
    }

    /* ---------- Cards ---------- */
    .card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        color: #94a3b8 !important;
        margin-bottom: 0.35rem;
    }
    .card-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f1f5f9 !important;
        line-height: 1.1;
    }
    .card-sub {
        font-size: 0.82rem;
        color: #94a3b8 !important;
        margin-top: 0.25rem;
    }

    /* ---------- Result banners ---------- */
    .result-danger {
        background: linear-gradient(120deg, #7f1d1d 0%, #b91c1c 100%);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 10px 30px rgba(185,28,28,0.35);
        margin-top: 1rem;
    }
    .result-safe {
        background: linear-gradient(120deg, #064e3b 0%, #047857 100%);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 10px 30px rgba(4,120,87,0.35);
        margin-top: 1rem;
    }
    .result-danger h2, .result-safe h2 {
        color: #ffffff !important;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .result-danger p, .result-safe p {
        color: #f1f5f9 !important;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] svg {
        fill: #94a3b8 !important;
    }

    /* ---------- Button ---------- */
    .stButton > button {
        width: 100%;
        padding: 0.85rem 1rem;
        font-size: 1.02rem;
        font-weight: 700;
        color: #ffffff !important;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        border: none;
        border-radius: 12px;
        box-shadow: 0 8px 22px rgba(124,58,237,0.35);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(124,58,237,0.5);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(148,163,184,0.15);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ---------- Dataframe ---------- */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Progress bar ---------- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "best_model.pkl"


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def model_is_loaded():
    return "model" in st.session_state


if not model_is_loaded():
    if not os.path.exists(MODEL_PATH):
        st.markdown(
            """
            <div class="hero">
                <h1>⚠️ Model Not Found</h1>
                <p>
                    The file <code>best_model.pkl</code> could not be located in the
                    application directory. Please make sure it is included in the
                    repository before deploying.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        st.session_state.model = load_model(MODEL_PATH)
    except Exception as e:
        st.error("❌ Unable to load the trained model.")
        st.code(str(e))
        st.info(
            "Make sure `best_model.pkl` exists and that the scikit-learn version "
            "used at training time matches the one installed here."
        )
        st.stop()


model = st.session_state.model

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>⚙️ AI-Based Predictive Maintenance</h1>
        <p>
            Predict whether an industrial machine is likely to fail based on real-time
            operating conditions. Fill in the sensor readings below and let the trained
            Gradient Boosting model estimate the failure risk.
        </p>
        <span class="badge">🤖 Gradient Boosting · 98.55% Accuracy</span>
        <span class="badge">📊 F1-Score 0.756</span>
        <span class="badge">✅ Model Ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📘 About the App")
    st.caption(
        "This tool uses a machine-learning pipeline trained on the AI4I 2020 "
        "Predictive Maintenance dataset to estimate machine failure probability."
    )

    st.divider()
    st.markdown("### 🎯 Model Metrics")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy", "98.55%")
        st.metric("Precision", "88.24%")
    with m2:
        st.metric("Recall", "66.18%")
        st.metric("F1-Score", "0.756")

    st.divider()
    st.markdown("### 🧪 Features Used")
    st.markdown(
        """
        - **Type** – Product quality (L/M/H)
        - **Air temperature** – Ambient temp (K)
        - **Process temperature** – Working temp (K)
        - **Rotational speed** – RPM
        - **Torque** – Newton-metres
        - **Tool wear** – Minutes used
        """
    )

    st.divider()
    st.caption("Built with Streamlit · scikit-learn · joblib")

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("## 🧾 Enter Machine Parameters")

with st.form("machine_input_form", clear_on_submit=False):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Machine Identity</div>', unsafe_allow_html=True)
        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            index=0,
            help="Product quality type: L = Low, M = Medium, H = High",
        )
        st.caption("Quality grade of the manufactured product.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Temperature Readings</div>', unsafe_allow_html=True)
        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=300.0,
            step=0.1,
            help="Ambient air temperature around the machine, in Kelvin.",
        )
        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=310.0,
            step=0.1,
            help="Temperature of the process itself, in Kelvin.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Mechanical Readings</div>', unsafe_allow_html=True)
        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=0,
            max_value=10000,
            value=1500,
            step=10,
            help="Machine rotational speed in revolutions per minute.",
        )
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=0.1,
            help="Force producing rotation, measured in Newton-metres.",
        )
        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=500,
            value=100,
            step=1,
            help="Accumulated tool usage in minutes.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Machine Failure")

# ============================================================
# PREDICTION
# ============================================================
if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )

    try:
        prediction = int(model.predict(input_data)[0])

        # Probabilities (all pipelines in this project expose predict_proba)
        try:
            proba = model.predict_proba(input_data)[0]
            failure_prob = float(proba[1]) * 100
        except Exception:
            failure_prob = None

        st.markdown("## 📊 Prediction Result")

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-danger">
                    <h2>⚠️ Machine Failure Predicted</h2>
                    <p>
                        The model has detected patterns consistent with an impending
                        failure. Immediate inspection and preventive maintenance are
                        strongly recommended for the machine before continuing operation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-safe">
                    <h2>✅ No Machine Failure Predicted</h2>
                    <p>
                        The machine is operating within the learned safe conditions.
                        Continue normal operations and keep monitoring sensor readings
                        periodically.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- Result metrics ----------
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Predicted Class</div>
                    <div class="card-value">{'Failure' if prediction == 1 else 'Healthy'}</div>
                    <div class="card-sub">Binary output from the model</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            if failure_prob is not None:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">Failure Probability</div>
                        <div class="card-value">{failure_prob:.2f}%</div>
                        <div class="card-sub">Model confidence estimate</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="card">
                        <div class="card-title">Failure Probability</div>
                        <div class="card-value">N/A</div>
                        <div class="card-sub">Not available</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with c3:
            risk_label = (
                "High Risk"
                if failure_prob is not None and failure_prob >= 50
                else "Medium Risk"
                if failure_prob is not None and failure_prob >= 25
                else "Low Risk"
            )
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Risk Level</div>
                    <div class="card-value">{risk_label}</div>
                    <div class="card-sub">Derived from failure probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- Probability bar ----------
        if failure_prob is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔎 Failure Probability")
            st.progress(min(max(failure_prob / 100, 0.0), 1.0))
            st.caption(
                f"The model estimates a **{failure_prob:.2f}%** chance of machine failure "
                f"under the provided operating conditions."
            )

        # ---------- Input summary ----------
        with st.expander("📋 View entered machine details", expanded=False):
            display_df = input_data.copy()
            display_df.insert(0, "Parameter", display_df.index + 1)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        st.error("❌ An error occurred while making the prediction.")
        st.code(str(e))
        st.caption(
            "Check the input column names and the preprocessing pipeline "
            "used when training the model."
        )
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc())

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.82rem; line-height:1.6;">
        ⚙️ <strong>AI-Based Predictive Maintenance</strong><br>
        Powered by Gradient Boosting &amp; scikit-learn · Built with Streamlit<br>
        <em>For informational purposes only — always confirm with physical inspection.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
