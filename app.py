import streamlit as st
import pandas as pd
import pickle

# Load trained model and preprocess
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
preprocess = pickle.load(open('preprocess.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

# Title
st.title('Medical Insurance Cost Prediction 💵🏥')

# Inputs
age = st.slider('Age', 18, 100, 30)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.slider('BMI', 10.0, 50.0, 25.0)
children = st.slider('Number of Children', 0, 5, 0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northeast', 'southeast', 'southwest', 'northwest'])

# Collect input into a raw dataframe
raw_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Preprocess input properly (this handles one-hot encoding etc)
X_input = preprocess.transform(raw_input)

# Predict
if st.button('Predict Medical Cost'):
    pred = rf_model.predict(X_input)[0]
    
    st.subheader('Prediction Result:')
    st.success(f'Estimated Annual Medical Cost: **${pred:,.2f}**')

    with st.expander('See raw input'):
        st.write(raw_input)
