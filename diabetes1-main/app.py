import os
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Custom CSS for background image
def set_background_image():
    # Specify the path to your background image
    background_image_path = os.path.join(os.path.dirname(__file__), 'background.jpg')
    # Generate CSS to set the background image
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url("data:image/jpeg;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Loading the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Displaying the app
set_background_image()  # Set the background image

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
