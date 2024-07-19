import time
import joblib
import pandas as pd
from config import *
import streamlit as st
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

def prediction_page():
    def convert_to_one_hot(selected_category, all_categories):
            one_hot = [True if category == selected_category else False for category in all_categories]
            for value in one_hot:
                user_input.append(value)

    def predict_alzheimer(input_data):
        loaded_model = joblib.load('diabetes_model.sav')
        predictions = loaded_model.predict(input_data)

        return predictions


# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))




# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
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