import os
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Getting the working directory of the app.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the model file
liver_model_path = os.path.join(working_dir, 'liver.sav')  # Adjust the model file name

# Load the pre-trained model
with open(liver_model_path, 'rb') as model_file:
    liver_model = pickle.load(model_file)

# Sample values for two different sets of data
sample1_values = {
    "Age": 65,
    "Gender": "Female",
    "Total_Bilirubin": 0.70,
    "Direct_Bilirubin": 0.10,
    "Alkaline_Phosphotase": 187,
    "Alamine_Aminotransferase": 16,
    "Aspartate_Aminotransferase": 18,
    "Total_Proteins": 6.80,
    "Albumin": 3.30,
    "Albumin_and_Globulin_Ratio": 0.90
}

sample2_values = {
    "Age": 25,
    "Gender": "Male",
    "Total_Bilirubin": 0.6,
    "Direct_Bilirubin": 0.1,
    "Alkaline_Phosphotase": 183,
    "Alamine_Aminotransferase": 91,
    "Aspartate_Aminotransferase": 53,
    "Total_Proteins": 5.5,
    "Albumin": 2.3,
    "Albumin_and_Globulin_Ratio": 0.7
}

# Select sample values based on the provided criteria
def select_sample_values(age, gender):
    if age == 65 and gender == "Female":
        return sample1_values
    elif age == 25 and gender == "Male":
        return sample2_values
    else:
        return None

st.title('Liver Disease Prediction using ML')

# Input fields
col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.number_input('Age', value=sample1_values['Age'])
with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'], index=0 if sample1_values['Gender'] == 'Male' else 1)
with col3:
    total_bilirubin = st.number_input('Total Bilirubin', value=sample1_values['Total_Bilirubin'])
with col4:
    direct_bilirubin = st.number_input('Direct Bilirubin', value=sample1_values['Direct_Bilirubin'])

with col1:
    alkaline_phosphotase = st.number_input('Alkaline Phosphotase', value=sample1_values['Alkaline_Phosphotase'])
with col2:
    alamine_aminotransferase = st.number_input('Alamine Aminotransferase', value=sample1_values['Alamine_Aminotransferase'])
with col3:
    aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', value=sample1_values['Aspartate_Aminotransferase'])
with col4:
    total_proteins = st.number_input('Total Proteins', value=sample1_values['Total_Proteins'])

with col1:
    albumin = st.number_input('Albumin', value=sample1_values['Albumin'])
with col2:
    albumin_and_globulin_ratio = st.number_input('Albumin and Globulin Ratio', value=sample1_values['Albumin_and_Globulin_Ratio'])

# Convert categorical variables to numerical values
gender_numeric = 1 if gender == 'Male' else 0

# Collect user inputs
user_input = [age, gender_numeric, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase,
              aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio]

liver_prediction = ''
if st.button('Liver Disease Test Result'):
    try:
        # Ensure all inputs are in the correct format
        user_input = [float(feature) for feature in user_input]

        # Make prediction
        liver_prediction = liver_model.predict([user_input])[0]
        
        # Determine if the person likely has liver disease based on gender and key indicators
        if (alamine_aminotransferase > 40 or aspartate_aminotransferase > 40 or total_bilirubin > 1.2 or direct_bilirubin > 0.3):
            liver_diagnosis = 'The person is likely to have Liver Disease'
        else:
            liver_diagnosis = 'The person is NOT likely to have Liver Disease'
        
        st.success(liver_diagnosis)
    except ValueError as ve:
        st.error(f'Please enter valid numbers for all fields. ValueError: {ve}')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

# Running the app
if __name__ == '__main__':
    st.success('Welcome to the Health Assistant application!')
