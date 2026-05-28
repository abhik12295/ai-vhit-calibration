import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="HIMP Gain Calibration Tool",
    layout="centered"
)

st.title("HIMP VOR Gain Calibration Tool")
st.write(
    "Clinical decision-support demo for proportional camera-placement inflation correction."
)

# ------------------------------------------------------------
# Load pattern calibration bundle
# ------------------------------------------------------------

model_bundle = joblib.load("himp_pattern_calibration_model.pkl")

pattern_lookup = model_bundle["pattern_lookup"]
pattern_table = model_bundle["pattern_table"]

inflated_patterns = model_bundle.get(
    "inflated_patterns",
    {
        "Right": ["RA", "LP"],
        "Left": ["LA", "RP"]
    }
)

# ------------------------------------------------------------
# Sidebar input
# ------------------------------------------------------------

st.sidebar.header("Patient / Test Input")

raw_gain = st.sidebar.number_input(
    "Raw HIMP VOR Gain",
    min_value=0.0,
    max_value=3.0,
    value=0.80,
    step=0.01
)

camera_side = st.sidebar.selectbox(
    "Camera Side",
    ["Right", "Left"]
)

canal = st.sidebar.selectbox(
    "Canal",
    ["RL", "LL", "RA", "LP", "LA", "RP"],
    index=2
)

canal_type = "Horizontal" if canal in ["RL", "LL"] else "Vertical"

pattern_key = (camera_side, canal)
pattern_found = pattern_key in pattern_lookup

# ------------------------------------------------------------
# Correction logic
# ------------------------------------------------------------

if pattern_found:
    learned_inflation_pct = pattern_lookup[pattern_key]
    learned_inflation_pct = max(learned_inflation_pct, 0)

    corrected_gain = raw_gain / (1 + learned_inflation_pct)
    correction_amount = raw_gain - corrected_gain

else:
    learned_inflation_pct = 0.0
    corrected_gain = raw_gain
    correction_amount = 0.0

distance_from_1 = abs(corrected_gain - 1.0)

# ------------------------------------------------------------
# Input summary
# ------------------------------------------------------------

st.subheader("Input Summary")

input_summary = pd.DataFrame([{
    "raw_himp_gain": raw_gain,
    "camera_side": camera_side,
    "canal": canal,
    "canal_type": canal_type,
    "pattern_found": pattern_found,
    "learned_inflation_pct": learned_inflation_pct
}])

st.dataframe(input_summary, use_container_width=True)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

st.subheader("Calibration Output")

col1, col2, col3 = st.columns(3)

col1.metric("Raw Gain", round(raw_gain, 3))
col2.metric("Learned Inflation %", f"{learned_inflation_pct * 100:.2f}%")
col3.metric("Corrected Gain", round(corrected_gain, 3))

col4, col5 = st.columns(2)

col4.metric("Correction Amount", round(correction_amount, 3))
col5.metric("Distance From 1.0", round(distance_from_1, 3))

st.subheader("Correction Formula")

st.code(
    "corrected_gain = raw_gain / (1 + learned_inflation_percentage)",
    language="text"
)

# ------------------------------------------------------------
# Pattern table display
# ------------------------------------------------------------

with st.expander("View learned camera-placement inflation patterns"):
    display_table = pattern_table.copy()
    display_table["learned_inflation_pct_display"] = (
        display_table["learned_inflation_pct"] * 100
    ).round(2)

    st.dataframe(display_table, use_container_width=True)

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

st.subheader("Clinical Interpretation")

if pattern_found:
    st.warning(
        "This canal-camera pair has a learned vertical camera-placement inflation pattern. "
        "The raw gain was proportionally corrected downward."
    )
else:
    st.info(
        "No learned inflation pattern exists for this canal-camera pair. "
        "No correction was applied."
    )

if corrected_gain < 0.70:
    st.error(
        "Corrected gain is below common lower-limit reference ranges. "
        "This may suggest possible vestibular hypofunction, but interpretation must be confirmed "
        "by a clinician using full vHIT traces, corrective saccades, symptoms, asymmetry, and clinical history."
    )
elif corrected_gain <= 1.20:
    st.success(
        "Corrected gain is within or near the expected reference region. "
        "Interpret together with symptoms, saccades, asymmetry, and clinical history."
    )
else:
    st.warning(
        "Corrected gain remains elevated after proportional calibration. "
        "Review camera placement, raw traces, and consider repeat testing."
    )

# ------------------------------------------------------------
# Limitation
# ------------------------------------------------------------

st.subheader("Important Limitation")

st.caption(
    "This model was trained only on healthy-subject HIMP data. "
    "It estimates camera-placement-related percentage inflation and does not diagnose vestibular disease."
)