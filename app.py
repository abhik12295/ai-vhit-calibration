import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="HIMP Gain Calibration Tool",
    layout="centered"
)

st.title("HIMP VOR Gain Calibration Tool")
st.write("Clinical decision-support demo for camera-placement bias correction.")

model_bundle = joblib.load("himp_ridge_calibration_model.pkl")

model = model_bundle["model"]
features = model_bundle["features"]
correction_strength = model_bundle["correction_strength"]

st.sidebar.header("Patient / Test Input")

raw_gain = st.sidebar.number_input(
    "Raw HIMP VOR Gain",
    min_value=0.0,
    max_value=3.0,
    value=1.30,
    step=0.01
)

camera_side = st.sidebar.selectbox(
    "Camera Side",
    ["Right", "Left"]
)

canal = st.sidebar.selectbox(
    "Canal",
    ["RL", "LL", "RA", "LP", "LA", "RP"]
)

canal_type = "Horizontal" if canal in ["RL", "LL"] else "Vertical"

inflated_pair_flag = int(
    ((camera_side == "Right") and (canal in ["RA", "LP"])) or
    ((camera_side == "Left") and (canal in ["LA", "RP"]))
)

input_df = pd.DataFrame([{
    "raw_himp_gain": raw_gain,
    "camera_side": camera_side,
    "canal": canal,
    "canal_type": canal_type,
    "inflated_pair_flag": inflated_pair_flag
}])

predicted_bias = model.predict(input_df[features])[0]

corrected_gain = raw_gain - correction_strength * predicted_bias
corrected_gain = max(0.65, min(1.35, corrected_gain))

distance_from_1 = abs(corrected_gain - 1.0)

st.subheader("Input Summary")

st.dataframe(input_df)

st.subheader("Calibration Output")

col1, col2, col3 = st.columns(3)

col1.metric("Raw Gain", round(raw_gain, 3))
col2.metric("Predicted Bias", round(predicted_bias, 3))
col3.metric("Corrected Gain", round(corrected_gain, 3))

st.metric("Distance From Healthy Center 1.0", round(distance_from_1, 3))

st.subheader("Interpretation")

if inflated_pair_flag == 1:
    st.warning(
        "This canal-camera combination is known to be at risk for artificial vertical canal gain inflation."
    )
else:
    st.info(
        "This canal-camera combination is not one of the primary expected inflated pairs."
    )

if corrected_gain < 0.70:
    st.error(
        "Corrected gain is below common lower-limit reference ranges. Clinician should consider possible vestibular hypofunction and confirm with full vHIT interpretation."
    )
elif corrected_gain <= 1.20:
    st.success(
        "Corrected gain is near the expected healthy range. Interpret together with symptoms, saccades, asymmetry, and clinical history."
    )
else:
    st.warning(
        "Corrected gain remains elevated after calibration. Review camera placement, traces, and repeat testing if needed."
    )

st.caption(
    "This tool is not a diagnostic device. It only estimates camera-placement-related HIMP gain bias."
)