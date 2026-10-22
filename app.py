
import streamlit as st
import pandas as pd
import joblib
import os
import traceback

# Page configuration
st.set_page_config(
    page_title="AI-Based Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

# Application title
st.title("AI-Based Predictive Maintenance")
st.write(
    "Predict whether a machine is likely to fail "
    "based on its operating conditions."
)

st.divider()

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

try:
    if not os.path.exists("best_model.pkl"):
        st.error(
            "Model file not found. "
            "Make sure best_model.pkl is in the GitHub repository."
        )
        st.stop()

    model = load_model()
    st.success("Trained model loaded successfully!")

except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.info(
        "Check that best_model.pkl exists and that the "
        "scikit-learn version matches the training environment."
    )
    st.stop()

# User inputs
st.subheader("Enter Machine Details")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temp = st.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.number_input(
    "Rotational Speed (rpm)",
    min_value=0,
    max_value=10000,
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=200.0,
    value=40.0,
    step=0.1
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=500,
    value=100
)

st.divider()

# Make prediction
if st.button("Predict Machine Failure", type="primary"):

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Prediction: Machine Failure")
            st.warning(
                "The model predicts a possible machine failure. "
                "Inspect the machine and its operating conditions."
            )
        else:
            st.success("✅ Prediction: No Machine Failure")
            st.write(
                "The model predicts no machine failure "
                "for the supplied operating conditions."
            )

        # Show input values used for prediction
        with st.expander("View entered machine details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.code(str(e))
        st.caption(
            "Check the input column names and the preprocessing "
            "pipeline used when training the model."
        )
