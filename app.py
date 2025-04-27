import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ---------- Load exported artefacts ----------
ROOT = Path(__file__).parent          # folder where app resides
preprocess    = pickle.load(open(ROOT / "preprocess.pkl",   "rb"))
feature_names = pickle.load(open(ROOT / "feature_names.pkl","rb"))
rf_model      = pickle.load(open(ROOT / "rf_model.pkl",     "rb"))

# ---------- Page config ----------
st.set_page_config(page_title="Medical Cost Predictor", page_icon="💵🏥")

st.title("Medical Insurance Cost Prediction 💵🏥")

# ---------- User inputs ----------
age      = st.slider ("Age", 18, 100, 30)
sex      = st.selectbox("Sex",     ["male", "female"])
bmi      = st.slider ("BMI", 10.0, 50.0, 25.0)
children = st.slider ("Number of children", 0, 5, 0)
smoker   = st.selectbox("Smoker",  ["yes", "no"])
region   = st.selectbox("Region",  ["northeast", "southeast", "southwest", "northwest"])

raw_input = pd.DataFrame(
    {
        "age":      [age],
        "sex":      [sex],
        "bmi":      [bmi],
        "children": [children],
        "smoker":   [smoker],
        "region":   [region],
    }
)

# ---------- Prediction ----------
if st.button("Predict Medical Cost"):
    X_input = preprocess.transform(raw_input)
    pred    = rf_model.predict(X_input)[0]

    st.subheader("Prediction result")
    st.write(f"💲 **Estimated annual charge:**  ${pred:,.2f}")
