import streamlit as st
import pickle
import numpy as np
st.title("Electricity Bill Calculator")
site=st.number_input("Enter the site area",0,30000)
structure_type=st.selectbox("Structure type",['Mixed-use', 'Residential', 'Commercial', 'Industrial'])

water_consumption = st.number_input(
    "Water Consumption (liters or m³)", 
    min_value=0.0, 
    step=0.1
)

recycling_rate = st.number_input(
    "Recycling Rate (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.1
)

utilisation_rate = st.number_input(
    "Utilisation Rate (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.1
)

air_quality_index = st.number_input(
    "Air Quality Index (AQI)",
    min_value=0,
    step=1
)

issue_resolution_time = st.number_input(
    "Issue Resolution Time (hours)",
    min_value=0.0,
    step=0.1
)

resident_count = st.number_input(
    "Resident Count",
    min_value=1,
    step=1
)
button=st.button("Predict")
if button:
    test=np.array([[site,water_consumption,recycling_rate,utilisation_rate,air_quality_index,issue_resolution_time,resident_count]])
with open("electricity.pkl","rb") as obj1:
    data=pickle.load(obj1)
test1=data["onehot"].transform([[structure_type]])
test2=np.hstack([test,test1])
scaled_test=data["scaler"].transform(test2)
result=data["model"].predict(scaled_test)[0]
st.success(f"Expected bill is {round(result)}")