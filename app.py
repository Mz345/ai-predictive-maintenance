import os
import traceback

import joblib
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
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(59,130,246,0.10), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(168,85,247,0.10), transparent 55%),
            #0b1220;
    }
    .main .block-container { max-width: 1180px; padding-top: 1.25rem; padding-bottom: 3rem; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #e5e7eb; }
    hr { border-color: rgba(148,163,184,0.15); }

    /* ---------- Hero ---------- */
    .hero {
        position: relative; padding: 2rem 2.2rem; border-radius: 20px;
        background: linear-gradient(135deg, #0b1220 0%, #111a33 40%, #1b1147 100%);
        border: 1px solid rgba(148,163,184,0.18); overflow: hidden; margin-bottom: 1.5rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.45);
    }
    .hero::before {
        content: ""; position: absolute; inset: -40% -20% auto auto;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(139,92,246,0.35), transparent 60%);
        filter: blur(20px); pointer-events: none;
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.6px; color: #f8fafc; }
    .hero p { margin: 0.6rem 0 1rem 0; max-width: 720px; font-size: 0.98rem; color: #cbd5e1; line-height: 1.6; }
    .pill {
        display: inline-flex; align-items: center; gap: 6px; margin-right: 8px;
        padding: 6px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        color: #e2e8f0; background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
    }
    .pill.green  { color:#6ee7b7; border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.10); }
    .pill.blue   { color:#93c5fd; border-color: rgba(59,130,246,0.35); background: rgba(59,130,246,0.10); }
    .pill.violet { color:#c4b5fd; border-color: rgba(139,92,246,0.35); background: rgba(139,92,246,0.10); }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 0.78rem; letter-spacing: 1.4px; font-weight: 700;
        text-transform: uppercase; color: #94a3b8; margin-bottom: 0.35rem;
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: linear-gradient(180deg, rgba(30,41,59,0.85), rgba(15,23,42,0.85));
        border: 1px solid rgba(148,163,184,0.16); border-radius: 14px;
        padding: 1rem 1.1rem; height: 100%; box-shadow: 0 6px 20px rgba(0,0,0,0.28);
    }
    .metric-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.1px; font-weight: 700; color: #94a3b8; }
    .metric-card .value { font-size: 1.55rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-top: 0.2rem; }
    .metric-card .sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; }

    /* ---------- Result banner ---------- */
    .result {
        border-radius: 16px; padding: 1.5rem 1.6rem; margin-top: 0.5rem;
        border: 1px solid transparent; box-shadow: 0 14px 40px rgba(0,0,0,0.35);
    }
    .result h2 { margin: 0; font-size: 1.5rem; font-weight: 800; color: #ffffff; }
    .result p  { margin-top: 0.5rem; color: #f1f5f9; opacity: 0.92; line-height: 1.55; font-size: 0.95rem; }
    .result.danger { background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 50%, #b91c1c 100%); border-color: rgba(239,68,68,0.45); }
    .result.safe   { background: linear-gradient(135deg, #052e2b 0%, #064e3b 50%, #047857 100%); border-color: rgba(16,185,129,0.45); }
    .result .risk  {
        display: inline-block; margin-top: 0.75rem; padding: 4px 10px; border-radius: 999px;
        font-size: 0.74rem; font-weight: 700; background: rgba(255,255,255,0.12);
        color: #ffffff; letter-spacing: 0.5px;
    }

    /* ---------- Component failure cards ---------- */
    .component-card {
        border-radius: 14px; padding: 1rem 1.1rem;
        border: 1px solid rgba(148,163,184,0.18);
        background: rgba(30,41,59,0.55);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        height: 100%;
    }
    .component-card.active {
        border-color: rgba(239,68,68,0.55);
        background: linear-gradient(180deg, rgba(127,29,29,0.55), rgba(69,10,10,0.35));
        box-shadow: 0 10px 26px rgba(239,68,68,0.22);
    }
    .component-card.inactive {
        border-color: rgba(16,185,129,0.35);
        background: linear-gradient(180deg, rgba(6,78,59,0.35), rgba(2,44,34,0.20));
    }
    .component-card .comp-title {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.95rem; font-weight: 700; color: #f1f5f9;
    }
    .component-card .comp-code {
        font-size: 0.7rem; letter-spacing: 1.1px; font-weight: 700;
        color: #94a3b8;
    }
    .component-card .comp-desc {
        font-size: 0.82rem; color: #cbd5e1; margin-top: 0.35rem; line-height: 1.4;
    }
    .component-card .comp-prob {
        margin-top: 0.6rem; font-size: 0.78rem; font-weight: 700;
        color: #f8fafc;
    }
    .component-card .status-dot {
        width: 10px; height: 10px; border-radius: 50%;
        display: inline-block;
    }
    .status-dot.red   { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
    .status-dot.green { background: #10b981; box-shadow: 0 0 8px #10b981; }

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
        width: 100%; padding: 0.9rem 1rem; font-weight: 700; font-size: 1rem;
        color: #ffffff !important;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        border: none; border-radius: 12px;
        box-shadow: 0 10px 25px rgba(124,58,237,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 14px 30px rgba(124,58,237,0.5);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background: rgba(11,18,32,0.97); border-right: 1px solid rgba(148,163,184,0.14); }
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
    .footer { text-align: center; color: #64748b; font-size: 0.8rem; line-height: 1.7; margin-top: 2.5rem; }
    .footer strong { color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
BINARY_MODEL_PATH = "best_model.pkl"
SUBTYPE_MODEL_PATH = "failure_subtype_models.pkl"
SUBTYPE_METRICS_PATH = "failure_subtype_metrics.pkl"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Human-readable metadata for each subtype
SUBTYPE_INFO = {
    "TWF": {
        "name": "Tool Wear Failure",
        "icon": "🔧",
        "desc": "The cutting tool has worn out beyond its safe limit.",
        "action": "Replace the cutting tool and check tool-change intervals.",
    },
    "HDF": {
        "name": "Heat Dissipation Failure",
        "icon": "🌡️",
        "desc": "Heat is not dissipating properly — temperature delta too low.",
        "action": "Inspect cooling fans, vents, and coolant flow.",
    },
    "PWF": {
        "name": "Power Failure",
        "icon": "⚡",
        "desc": "Power delivered is outside the safe operating band.",
        "action": "Check electrical supply, wiring, and power ratings.",
    },
    "OSF": {
        "name": "Overstrain Failure",
        "icon": "🛠️",
        "desc": "Mechanical overstrain from excessive load × tool wear.",
        "action": "Reduce load/torque or replace worn tool.",
    },
    "RNF": {
        "name": "Random Failure",
        "icon": "🎲",
        "desc": "Historically unpredictable failure — low model confidence expected.",
        "action": "Inspect the machine physically; not reliably predictable.",
    },
}


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models(binary_path: str, subtype_path: str, metrics_path: str):
    binary = joblib.load(binary_path) if os.path.exists(binary_path) else None
    subtypes = joblib.load(subtype_path) if os.path.exists(subtype_path) else None
    metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}
    return binary, subtypes, metrics


def load_models_or_die():
    if "bundle" in st.session_state:
        return st.session_state.bundle

    if not os.path.exists(BINARY_MODEL_PATH):
        st.markdown(
            """
            <div class="hero">
                <h1>⚠️ Model file not found</h1>
                <p>
                    <code>best_model.pkl</code> is missing. Please include it in the
                    repository before deploying.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        binary, subtypes, metrics = load_models(
            BINARY_MODEL_PATH, SUBTYPE_MODEL_PATH, SUBTYPE_METRICS_PATH
        )
    except Exception as exc:
        st.error("❌ Unable to load trained models.")
        st.code(str(exc))
        st.stop()

    st.session_state.bundle = {"binary": binary, "subtypes": subtypes, "metrics": metrics}
    return st.session_state.bundle


bundle = load_models_or_die()
binary_model = bundle["binary"]
subtype_models = bundle["subtypes"]
subtype_metrics = bundle["metrics"]


# ============================================================
# HELPERS
# ============================================================
def predict_binary(input_df: pd.DataFrame):
    prediction = int(binary_model.predict(input_df)[0])
    prob = None
    if hasattr(binary_model, "predict_proba"):
        try:
            prob = float(binary_model.predict_proba(input_df)[0][1])
        except Exception:
            prob = None
    return prediction, prob


def predict_subtypes(input_df: pd.DataFrame) -> dict:
    """Return {subtype_code: probability_of_that_failure}."""
    if not subtype_models:
        return {}

    results = {}
    for code, mdl in subtype_models.items():
        try:
            prob = float(mdl.predict_proba(input_df)[0][1])
        except Exception:
            prob = 0.0
        results[code] = prob
    return results


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


def render_component_card(code: str, prob: float, threshold: float = 0.5):
    info = SUBTYPE_INFO[code]
    active = prob >= threshold
    dot_class = "red" if active else "green"
    card_class = "active" if active else "inactive"
    status = "FAULT DETECTED" if active else "OK"

    st.markdown(
        f"""
        <div class="component-card {card_class}">
            <div class="comp-title">
                <span class="status-dot {dot_class}"></span>
                {info['icon']} {info['name']}
            </div>
            <div class="comp-code">{code} · {status}</div>
            <div class="comp-desc">{info['desc']}</div>
            <div class="comp-prob">Failure probability: {prob*100:.1f}%</div>
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
            Estimate whether a machine is likely to fail from its live operating
            conditions — and if it is likely to fail, <strong>which component</strong>
            is the culprit. Powered by a <strong>Gradient Boosting</strong> pipeline
            with a multi-label failure-type analyzer.
        </p>
        <span class="pill green">● Models ready</span>
        <span class="pill blue">98.55% Accuracy</span>
        <span class="pill violet">5 Failure Subtypes</span>
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
        "Two-stage AI: first predicts **if** the machine will fail, then identifies "
        "**which component** is most likely to cause the failure."
    )

    st.divider()
    st.markdown("### 🎯 Binary model performance")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", "98.55%")
    m1.metric("Precision", "88.24%")
    m2.metric("Recall", "66.18%")
    m2.metric("F1-Score", "0.756")

    if subtype_metrics:
        st.divider()
        st.markdown("### 🧩 Subtype models (F1)")
        for code, m in subtype_metrics.items():
            info = SUBTYPE_INFO[code]
            st.markdown(
                f"- **{info['icon']} {code}** ({info['name']}): "
                f"`{m.get('f1', 0):.3f}`"
            )

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
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section-label">Machine</div>', unsafe_allow_html=True)
        machine_type = st.selectbox(
            "Type", ["L", "M", "H"], index=0,
            help="Product quality tier: L = Low, M = Medium, H = High.",
        )
        st.markdown('<div class="section-label" style="margin-top:1rem;">Temperature</div>',
                    unsafe_allow_html=True)
        air_temp = st.number_input("Air temperature (K)", 250.0, 350.0, 300.0, 0.1)
        process_temp = st.number_input("Process temperature (K)", 250.0, 400.0, 310.0, 0.1)

    with right:
        st.markdown('<div class="section-label">Mechanics</div>', unsafe_allow_html=True)
        rotational_speed = st.number_input("Rotational speed (rpm)", 0, 10000, 1500, 10)
        torque = st.number_input("Torque (Nm)", 0.0, 200.0, 40.0, 0.1)
        tool_wear = st.number_input("Tool wear (min)", 0, 500, 100, 1)

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
        # ---------- Stage 1: will it fail? ----------
        prediction, failure_prob = predict_binary(input_df)
        st.markdown("## 📊 Prediction")

        if prediction == 1:
            risk = risk_level(failure_prob)
            prob_txt = (f"Estimated failure probability: <strong>{failure_prob*100:.2f}%</strong>."
                        if failure_prob is not None else "")
            st.markdown(
                f"""
                <div class="result danger">
                    <h2>⚠️ Machine Failure Predicted</h2>
                    <p>
                        The model detected operating conditions consistent with a
                        potential failure. See the component breakdown below.
                        {prob_txt}
                    </p>
                    <span class="risk">Risk level: {risk}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            risk = risk_level(failure_prob)
            prob_txt = (f"Estimated failure probability: <strong>{failure_prob*100:.2f}%</strong>."
                        if failure_prob is not None else "")
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
            render_metric("Predicted Class",
                          "Failure" if prediction == 1 else "Healthy",
                          "Binary output from the model")
        with k2:
            render_metric("Failure Probability",
                          f"{failure_prob*100:.2f}%" if failure_prob is not None else "N/A",
                          "Model confidence estimate")
        with k3:
            render_metric("Risk Level", risk, "Derived from probability")

        # ---------- Probability bar ----------
        if failure_prob is not None:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Failure probability</div>',
                        unsafe_allow_html=True)
            st.progress(min(max(failure_prob, 0.0), 1.0))

        # ---------- Stage 2: component breakdown ----------
        if subtype_models:
            st.markdown("## 🧩 Component Failure Analysis")
            st.caption(
                "Independent probability of each failure subtype, estimated by "
                "dedicated Gradient Boosting classifiers trained on the AI4I 2020 dataset."
            )

            subtype_probs = predict_subtypes(input_df)

            # Sort so the highest-risk component appears first
            sorted_codes = sorted(subtype_probs, key=lambda c: subtype_probs[c], reverse=True)

            cols = st.columns(2)
            for i, code in enumerate(sorted_codes):
                with cols[i % 2]:
                    render_component_card(code, subtype_probs[code])
                if i == 3:
                    # Reopen a fresh row after 4 cards
                    cols = st.columns(2)

            # Most likely cause
            top_code = sorted_codes[0]
            top_prob = subtype_probs[top_code]
            if top_prob >= 0.5:
                top = SUBTYPE_INFO[top_code]
                st.markdown(
                    f"""
                    <div class="result danger" style="margin-top:1rem;">
                        <h2>{top['icon']} Most likely fault: {top['name']}</h2>
                        <p>
                            <strong>Code:</strong> {top_code} ·
                            <strong>Probability:</strong> {top_prob*100:.1f}%<br>
                            <strong>Recommended action:</strong> {top['action']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result safe" style="margin-top:1rem;">
                        <h2>✅ No specific component fault flagged</h2>
                        <p>
                            Although the binary model signalled a possible failure,
                            no single subtype exceeded the 50% confidence threshold.
                            Consider a manual inspection.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Detailed probability breakdown (numeric)
            with st.expander("📈 Full probability breakdown"):
                rows = []
                for code, prob in subtype_probs.items():
                    info = SUBTYPE_INFO[code]
                    rows.append({
                        "Code": code,
                        "Component": info["name"],
                        "Probability": f"{prob*100:.2f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        else:
            st.info(
                "ℹ️ **Component-level analysis is not available.** "
                "Add `failure_subtype_models.pkl` to the repository to enable it."
            )

        # ---------- Input summary ----------
        with st.expander("📋 View entered parameters"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}),
                         use_container_width=True)

    except Exception as exc:
        st.error("❌ An error occurred while making the prediction.")
        st.code(str(exc))
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc())

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        <strong>AI-Based Predictive Maintenance</strong><br>
        Gradient Boosting · Multi-label component analysis · Streamlit<br>
        <em>For informational purposes only — always confirm with physical inspection.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
