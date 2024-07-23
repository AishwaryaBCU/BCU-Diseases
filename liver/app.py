import os
import base64
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🩺")

# Function to set background image
def set_page_background(image_path):
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
background_image_path = 'liver/bg.webp'
set_page_background(background_image_path)

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

def home_page():
    st.title("Predictor that You can Trust")
    st.write("""
    **Liver Disease** is a condition characterized by the dysfunction or damage to the liver, resulting in impaired liver function.
    Early detection and intervention are crucial to prevent further liver damage and complications.
    """)
    st.subheader("Machine Learning Project")
    st.write("""
    This liver disease prediction tool is developed using machine learning techniques to predict the likelihood of liver disease in individuals based on various health parameters. The model is trained on a dataset that includes parameters such as:
    
    - Age
    - Alkaline Phosphotase
    - Albumin
    - Gender
    - Alamine Aminotransferase
    - Albumin and Globulin Ratio
    - Total Bilirubin
    - Aspartate Aminotransferase
    - Direct Bilirubin
    - Total Proteins
    
    **Model Accuracy**: The machine learning model used in this project achieves an accuracy of approximately 71% on the test dataset. This means that the model is able to correctly predict liver disease approximately 71% of the time based on the given health parameters.
    """)
    st.write("Go to the 'Liver Disease Prediction' page to use the online predictor.")

def disclaimer_page():
    st.title("Disclaimer")
    st.write("""
    The predictions made by this application are based on machine learning models and should not be considered as medical advice.
    Always consult with a healthcare professional for medical advice, diagnosis, or treatment.
    """)

def liver_disease_prediction():
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

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Liver Disease Prediction", "Disclaimer"])

if page == "Home":
    home_page()
elif page == "Liver Disease Prediction":
    liver_disease_prediction()
elif page == "Disclaimer":
    disclaimer_page()

# Running the app
if __name__ == '__main__':
    st.success('Welcome to the Health Assistant application!')
