import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Custom CSS for background image
def set_background_image():
    # Specify the path to your background image
    background_image_path = 'background.jpg'
    # Generate CSS to set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("BCU-Diseases/diabetes1-main/background.jpg");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
set_background_image()
# Getting the directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define the absolute path to your model file
model_file_path = os.path.join(working_dir, 'diabetes_model.sav')

# Loading the saved model
try:
    with open(model_file_path, 'rb') as f:
        diabetes_model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Could not find the file {model_file_path}. Make sure the file exists.")

# Displaying the app
st.title('Diabetes Prediction using ML')

col1, col2, col3 = st.columns(3)
with col1:
    Pregnancies = st.text_input('Number of Pregnancies')
with col2:
    Glucose = st.text_input('Glucose Level')
with col3:
    BloodPressure = st.text_input('Blood Pressure value')
with col1:
    SkinThickness = st.text_input('Skin Thickness value')
with col2:
    Insulin = st.text_input('Insulin Level')
with col3:
    BMI = st.text_input('BMI value')
with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
with col2:
    Age = st.text_input('Age of the Person')

diab_diagnosis = ''
if st.button('Diabetes Test Result'):
    try:
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                      float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
        st.success(diab_diagnosis)
    except ValueError:
        st.error('Please enter valid numbers for all fields.')

# Running the app
if __name__ == '__main__':
    st.success('Welcome to the Health Assistant application!')
