import pandas as pd
import joblib
import streamlit as st

model = joblib.load("sales_model.pkl")
encoder = joblib.load("sales_encoder.pkl")

region = st.text_input("what is your region north/south: ")
age = st.number_input("how old are you?: ")
gender = st.selectbox("what is your gender: ", ['male', 'female'])
budget = st.number_input("how much is your budget: ")
previous_purchase = st.text_input("what did you bought last: ")

if st.button("Recommend"):
   sample_data = pd.DataFrame({
                           "region": ['region'],
                           "age": [age],
                           "gender": ['gender'],
                           "budget": [budget],
                           "previous_purchase": ['previous_purchase']
   })
   sample_data['region'] = sample_data['region'].str.lower()
   sample_data['gender'] = sample_data['gender'].str.lower()
   sample_data['previous_purchase'] = sample_data['previous_purchase'].str.lower()

   converted = encoder.transform(sample_data)
   make_recommendation = model.predict(converted)
   st.success(f"Recommended product: {make_recommendation}")


   