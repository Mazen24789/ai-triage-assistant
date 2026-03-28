import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="AI Triage Assistant", page_icon="🚨")

# 🔥 Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🚨 AI Emergency Triage Assistant")
st.write("Predict patient urgency using vital signs in real-time.")
st.info("⚡ Helps prioritize emergency patients faster and reduce critical delays.")

st.write("")

# 🔥 Demo scenarios
st.subheader("Demo Scenarios")

scenario = st.selectbox(
    "Select Patient Case",
    ["Custom Input", "Critical Patient", "Stable Patient"]
)

st.write("")

# 🔥 Patient input
st.subheader("Patient Vitals Input")

if scenario == "Critical Patient":
    heart_rate = 140
    temperature = 39.5
    o2 = 82
elif scenario == "Stable Patient":
    heart_rate = 75
    temperature = 36.8
    o2 = 98
else:
    heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 90)
    temperature = st.slider("Temperature (°C)", 35.0, 42.0, 37.0)
    o2 = st.slider("Oxygen Saturation (%)", 70, 100, 98)

# 🔥 Chief complaint
symptom = st.text_input("Chief Complaint", "Shortness of breath")

st.write("")

if st.button("Predict", use_container_width=True):

    # 🔥 REAL MODEL PREDICTION
    X = np.array([[heart_rate, temperature, o2]])
    pred = int(model.predict(X)[0])

    st.subheader("Prediction Result")

    if pred <= 2:
        st.error(f"🔴 HIGH RISK — CTAS {pred}")
    elif pred == 3:
        st.warning(f"🟡 MEDIUM RISK — CTAS {pred}")
    else:
        st.success(f"🟢 LOW RISK — CTAS {pred}")

    # 🔥 Critical alert (safety rule)
    if o2 < 90 or heart_rate > 120:
        st.error("🚨 CRITICAL CONDITION — Immediate Attention Needed")

    # 🔥 Clinical explanation
    st.write("### Clinical Insight")

    if pred <= 2:
        st.write("Patient shows signs of instability. Immediate medical attention required.")
    elif pred == 3:
        st.write("Moderate condition. Patient should be monitored closely.")
    else:
        st.write("Vitals are within normal range. Lower urgency.")

    # 🔥 Confidence bar
    st.progress(90 if pred <= 2 else 70 if pred == 3 else 50)

    st.write("Chief Complaint:", symptom)