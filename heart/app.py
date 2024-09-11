import os
import base64
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Function to set background image
def set_page_background(image_path):
    @st.cache_data
    def get_base64_of_bin_file(filename):
        with open(filename, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    if os.path.exists(image_path):
        bin_str = get_base64_of_bin_file(image_path)
        page_bg_img = f'''
            <style>
            .stApp {{
                background-image: url("data:image/webp;base64,{bin_str}");
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
background_image_path = 'heart/bg.webp'
set_page_background(background_image_path)

# Sidebar setup and content
st.sidebar.title("Heart Disease Prediction")
app_mode = st.sidebar.selectbox(
    "Please navigate through the different sections",
    ["Home", "Heart Disease Prediction", "Disclaimer"]
)

st.sidebar.write("""
# Disclaimer
The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.
""")

# Function to load saved model
@st.cache_data
def load_model(model_file):
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error(f"Model file '{model_file}' not found. Please ensure the file is in the correct directory.")
        st.stop()

# Load model
heart_disease_model_path = 'heart/heart_disease_model.sav'
heart_disease_model = load_model(heart_disease_model_path)

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
        fbs = st.selectbox("Fasting blood sugar level (1 = >120 mg/dL, 0 = <=120 mg/dL)", [0, 1])
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
        st.title("Welcome to Heart Disease Prediction System")
        st.write("""
        ## About Heart Disease
        Heart disease describes a range of conditions that affect your heart. These diseases include blood vessel diseases, such as coronary artery disease, heart rhythm problems (arrhythmias), and heart defects you're born with (congenital heart defects), among others.

        ## Machine Learning Project
        This heart disease prediction tool is developed using machine learning techniques to predict the likelihood of heart disease in individuals based on various health parameters. The model is trained on the Cleveland Heart Disease Dataset, which includes parameters such as:
        - Age
        - Resting Blood Pressure
        - Resting Electrocardiographic results
        - ST depression induced by exercise
        - Thalassemia (thal): 0 = normal; 1 = fixed defect; 2 = reversible defect
        - Sex
        - Serum Cholesterol in mg/dl
        - Maximum Heart Rate achieved
        - Slope of the peak exercise ST segment
        - Chest Pain types
        - Fasting Blood Sugar > 120 mg/dl
        - Exercise Induced Angina
        - Major vessels colored by fluoroscopy

        ### Model Accuracy
        The machine learning model used in this project achieves an accuracy of approximately 85% on the test dataset. This means that the model is able to correctly predict heart disease 85% of the time based on the given health parameters.

        ## Go to Online Predictor
        """)
    elif app_mode == "Heart Disease Prediction":
        show_heart_disease_prediction()
    elif app_mode == "Disclaimer":
        st.title("Disclaimer")
        st.write("""
        The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.
        """)

if __name__ == "__main__":
    main()
