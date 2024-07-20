import os
import base64
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Prediction System", page_icon="🩺", layout="wide")

# Function to set background image
def set_page_background(image_path):
    @st.cache(suppress_st_warning=True)
    def get_base64_of_bin_file(filename):
        with open(filename, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    if os.path.exists(image_path):
        bin_str = get_base64_of_bin_file(image_path)
        page_bg_img = f'''
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{bin_str}");
                background-size: cover;
            }}
            </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        st.warning(f"Background image file '{image_path}' not found.")
        st.text(f"Current working directory: {os.getcwd()}")
        st.text(f"Contents of the current directory: {os.listdir(os.getcwd())}")

# Set background image path
background_image_path = 'diabetes/background.jpg'
set_page_background(background_image_path)

# Sidebar setup and content
st.sidebar.title("Health Prediction System")
app_mode = st.sidebar.selectbox(
    "Please navigate through the different sections",
    ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Disclaimer"]
)

st.sidebar.write("""
# Disclaimer
The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.
""")

# Function to load saved model
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

# Load models
diabetes_model_path = 'diabetes_model.sav'
heart_disease_model_path = 'heart_disease_model.sav'
diabetes_model = load_model(diabetes_model_path)
heart_disease_model = load_model(heart_disease_model_path)

def show_diabetes_prediction():
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

    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
            else:
                st.success('The person is not diabetic')
        except ValueError:
            st.error('Please enter valid numbers for all fields.')

def show_heart_disease_prediction():
    st.title("Heart Disease Risk Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age ", step=1, min_value=1)
        trestbps = st.number_input("Resting Blood Pressure", step=1, min_value=1)
        restecg = st.selectbox("Resting Electrocardiographic Results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable abnormality)", [0, 1, 2])
        oldpeak = st.number_input("ST Depression Induced by Exercise")
        thal = st.selectbox("Thal", [1, 2, 3])

    with col2:
        sex = st.selectbox("Sex (1 = male, 0 = female)", [0, 1])
        chol = st.number_input("Serum Cholestoral in mg/dl", step=1, min_value=1)
        thalach = st.number_input("Maximum Heart Rate Achieved", step=1, min_value=1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])

    with col3:
        cp = st.selectbox("Chest Pain Type (0= typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting blood sugar level (1 = >120 mg/dL, 0 = <=120 mg/dL).", [0, 1])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        ca = st.selectbox("Major Vessels Colored by Flourosopy", [0, 1, 2, 3])

    if st.button("Predict Heart Disease Risk"):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            prediction = heart_disease_model.predict([user_input])
            if prediction[0] == 1:
                st.success("The person has a risk of heart disease.")
            else:
                st.success("The person does not have a risk of heart disease.")
        except ValueError:
            st.error('Please enter valid numbers for all fields.')

def main():
    if app_mode == "Home":
        st.title("Welcome to Health Prediction System")
        st.write("""
        ## About the System
        This system provides tools for predicting diabetes and heart disease risk using machine learning models. Use the sidebar to navigate to the respective prediction sections.

        ### Disclaimer
        The predictions provided by this system are for informational purposes only and are not a substitute for professional medical advice. Consult a healthcare professional for accurate diagnosis and advice.
        """)
    elif app_mode == "Diabetes Prediction":
        show_diabetes_prediction()
    elif app_mode == "Heart Disease Prediction":
        show_heart_disease_prediction()
    elif app_mode == "Disclaimer":
        st.title("Disclaimer")
        st.write("""
        The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.
        """)

if __name__ == "__main__":
    main()
