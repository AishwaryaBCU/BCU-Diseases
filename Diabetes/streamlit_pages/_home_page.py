import streamlit as st
from config import BANNER

def home_page():
    st.image(BANNER)
    st.write("""
       
        ## About Diabetes
        Diabetes is a chronic medical condition that affects how your body turns food into energy. It occurs when your blood glucose, also called blood sugar, is too high. Over time, having too much glucose in your blood can cause serious health problems. The most common types of diabetes are Type 1, Type 2, and gestational diabetes.

        ### Machine Learning Project
        This diabetes prediction tool is developed using machine learning techniques to predict the likelihood of diabetes in individuals based on various health parameters. The model is trained on the Pima Indians Diabetes Dataset, which includes parameters such as:

        - Number of Pregnancies
        - Glucose Level
        - Blood Pressure
        - Skin Thickness
        - Insulin Level
        - Body Mass Index (BMI)
        - Diabetes Pedigree Function
        - Age

        ### Model Accuracy
        The machine learning model used in this project achieves an accuracy of approximately 80% on the test dataset. This means that the model is able to correctly predict diabetes 80% of the time based on the given health parameters.
        
        <br>
    """, unsafe_allow_html=True)

    st.caption('Finished reading? Navigate to the `Prediction Page` to make some predictions.')
