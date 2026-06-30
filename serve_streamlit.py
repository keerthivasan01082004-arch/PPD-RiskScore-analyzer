import sys
sys.path.append('.')

import streamlit as st
import pandas as pd
from src.config import FEATURE_COLS, ENCODING_MAP, MODEL_PATHS
from src.utils import load_model
from joblib import load

st.set_page_config(page_title="PPD Risk Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 PPD Risk Screening Chatbot")
st.markdown("Answer the following questions (Yes / No / Sometimes).")

def encode_choice(choice):
    return ENCODING_MAP[choice]

with st.form("ppd_form"):
    inputs = {}
    for col in FEATURE_COLS:
        choice = st.radio(
            f"{col.replace('_',' ').title()}",
            options=list(ENCODING_MAP.keys()),
            index=1
        )
        inputs[col] = encode_choice(choice)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    X = pd.DataFrame([inputs])
    model = load_model(MODEL_PATHS["lightgbm"])
    le = load("models/label_encoder.pkl")

    probs = model.predict_proba(X)[0]
    idx = probs.argmax()
    label_str = le.inverse_transform([idx])[0]

    st.subheader("Result")
    st.success(f"Predicted Risk Level: **{label_str}**")
    st.write(f"Class probabilities (Low/Moderate/High): {probs}")

    suicidal = inputs.get("suicidal_ideation", 0)
    if label_str == "Low":
        rec = "Low risk. Focus on healthy routines and self-care."
    elif label_str == "Moderate":
        rec = "Moderate risk. Counseling and family support recommended."
    else:
        if suicidal == 2:
            rec = "High risk with suicidal thoughts. Seek immediate professional help."
        else:
            rec = "High risk. Please consult a mental health professional promptly."

    st.subheader("Recommendations")
    st.warning(rec)
