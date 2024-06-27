import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from style_css import style

st.set_page_config(
    page_title="Scenario Analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

style()

df =pd.read_csv("health_workers.csv")

# Load the models
model_doctors = joblib.load("model_doctors.pkl")
model_nurses = joblib.load("model_nurses.pkl")

# Streamlit application
st.write("## Healthcare Workforce Scenario Analysis")

st.write(
    """
    #### What is Health Workforce Density?
    Health Workforce Density means the number of healthcare workers (like doctors, nurses, and midwives) available to take care of people in a given area. It's measured by counting how many of these workers are available for every 10,000 people in the population.

    #### Why is it Important?
    Imagine you're in a town, and there are only a few doctors, nurses, and midwives available. If many people get sick at the same time, there won't be enough healthcare workers to take care of everyone quickly and effectively. Having an adequate number of healthcare workers ensures that people can get medical help when they need it, without long waits or shortages.

    #### How is it Measured?
    We look at how many doctors, nurses, and midwives there are for every 10,000 people. This gives us a good idea of whether there are enough healthcare workers to meet the needs of the community.

    #### WHO Recommendations
    The World Health Organization (WHO) gives us some guidelines on the minimum number of healthcare workers needed to ensure good healthcare services:

    Doctors: At least 1 doctor for every 1,000 people. This means for every 10,000 people, there should be at least 10 doctors.
    Nurses: At least 3 nurses for every 1,000 people. So, for every 10,000 people, there should be at least 30 nurses.
   
    #### Overall Health Workers:
    To achieve good healthcare coverage, there should be at least 44.5 health workers (including doctors and nurses) for every 10,000 people.
    """
)


# Function to calculate required healthcare workers
def calculate_additional_workers(df, population_increase=0):
    WHO_DOCTORS_PER_10000 = 10
    WHO_NURSES_PER_10000 = 30
    
    df = df.copy()
    
    # Apply population increase
    df['Population'] = df['Population'] * (1 + population_increase / 100)
    
    # Prepare the data for prediction
    X = df[['Doctors', 'Nurses', 'Population']]
    
    # Predict deficits using the models
    df['Predicted Deficit Doctors'] = model_doctors.predict(X)
    df['Predicted Deficit Nurses'] = model_nurses.predict(X)

    df['Predicted Deficit Doctors'] = df['Predicted Deficit Doctors'].astype(int)
    df['Predicted Deficit Nurses'] =  df['Predicted Deficit Nurses'].astype(int) * 3

    # Calculate the additional doctors and nurses needed
    df['Additional Doctors'] = np.round((df['Predicted Deficit Doctors'] / WHO_DOCTORS_PER_10000 * df['Population'] / 10000))
    df['Additional Nurses'] = np.round((df['Predicted Deficit Nurses'] / WHO_NURSES_PER_10000 * df['Population'] / 10000))
    
    return df[['LGA', 'Population', 'Predicted Deficit Doctors', 'Predicted Deficit Nurses', 'Additional Doctors', 'Additional Nurses']]


st.sidebar.header("Input Parameters")

# User input for population increase percentage
population_increase = st.sidebar.slider("Increase in Population (%)", min_value=0, max_value=100, value=0, step=5)


# Calculate the additional healthcare workers needed
result_df = calculate_additional_workers(df, population_increase)

result_df[['Current Doctors', 'Current Nurse']] = df[['Doctors','Nurses']]

result_df = result_df[['LGA', 'Current Doctors','Current Nurse','Population','Predicted Deficit Doctors', 'Predicted Deficit Nurses', 'Additional Doctors','Additional Nurses']]

st.dataframe(result_df.set_index("LGA"))

# Save result to CSV
csv = result_df.to_csv(index=False)
st.download_button(label="Download Results as CSV", data=csv, file_name="scenario_analysis_results.csv", mime="text/csv")
