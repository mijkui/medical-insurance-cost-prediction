import streamlit as st, pandas as pd, joblib, numpy as np

# configure page title and icon
st.set_page_config(page_title="Medical Cost Predictor", page_icon="💲")
st.title("💲 Medical Insurance Cost Estimator")

# load the saved model pipeline
model = joblib.load("model/medical_cost.pkl")

# — build a simple form for user input —
with st.form("predict"):
    age     = st.number_input("Age", 18, 100, 30)
    sex     = st.selectbox("Sex", ["male", "female"])
    bmi     = st.number_input("BMI", 15.0, 50.0, 28.0, step=0.1)
    children= st.number_input("Number of Children", 0, 5, 0)
    smoker  = st.selectbox("Smoker", ["yes", "no"])
    region  = st.selectbox("Region", ["southwest","southeast",
                                      "northwest","northeast"])
    submit  = st.form_submit_button("Predict")

# when the user clicks Predict…
if submit:

     # assemble a DataFrame with the inputs
    df = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region
    }])

    # make prediction, extract the single float
    cost = model.predict(df)[0]

     # display result nicely
    st.success(f"Estimated annual medical cost: **${cost:,.0f}**")