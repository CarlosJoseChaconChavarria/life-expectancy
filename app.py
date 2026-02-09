import streamlit as st # The tool that builds the website
import pandas as pd
import joblib
import numpy as np

# 1. LOAD THE SAVED BRAIN
model = joblib.load('life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. DESIGN THE WEBSITE UI
st.title("🌍 Life Expectancy Predictor")
st.write("Enter the details below to predict the life expectancy of a country.")

# 3. CREATE INPUT BOXES FOR THE USER
# We ask the user for the 4 things our model learned from
mortality = st.number_input("Adult Mortality (per 1000)", min_value=0, max_value=1000, value=200)
alcohol = st.number_input("Alcohol Consumption (liters per capita)", min_value=0.0, max_value=20.0, value=5.0)
gdp = st.number_input("GDP of the country", min_value=1, max_value=150000, value=5000)
schooling = st.number_input("Average Years of Schooling", min_value=0.0, max_value=20.0, value=10.0)

# 4. THE "PREDICT" BUTTON
if st.button("Predict Life Expectancy"):
    # A. Transform GDP (the 'Shrink Ray' we used in Step 3)
    gdp_log = np.log1p(gdp)
    
    # B. Scale the inputs (the 'Squasher' from Step 4)
    # The brain expects numbers between 0 and 1
    inputs = pd.DataFrame([[mortality, alcohol, gdp_log, schooling]], 
                          columns=['Adult Mortality', 'Alcohol', 'GDP_transformed', 'Schooling'])
    inputs_scaled = scaler.transform(inputs)
    
    # C. Make the final guess
    prediction = model.predict(inputs_scaled)
    
    # D. Show the result!
    st.success(f"The predicted Life Expectancy is: {prediction[0]:.2f} years")