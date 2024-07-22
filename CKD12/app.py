import os
import pickle
import streamlit as st
import base64
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Function to set the background image
def set_page_background(image_path):
    @st.cache_data
    def get_base64_of_bin_file(filename):
        with open(filename, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Convert image to base64
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

# Get current working directory and file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
background_image_path = os.path.join(current_dir, 'bg.jpg')  # Update this to the path of your background image

# Set background image
set_page_background(background_image_path)

# Add custom CSS to change font and background colors
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f0;  /* Light gray background */
    }
    .stTitle, .stHeader, .stMarkdown, .stText {
        color: #333;  /* Dark text color for visibility */
    }
    .stMarkdown {
        background-color: rgba(255, 255, 255, 0.7);  /* Slightly transparent white background for text blocks */
        padding: 15px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title('👨‍⚕️ Chronic Kidney Disease (CKD) Predictor')

# Description
st.markdown("""
    <div class="stMarkdown">
    **Chronic Kidney Disease (CKD)** is a condition where your kidneys don't work as well as they should for a long time. It can make you feel tired, swollen, or have trouble thinking clearly. This web app predicts if a patient has **Chronic Kidney Disease (CKD)** based on the patient's data.
    </div>
""", unsafe_allow_html=True)

# Path to the model file and other assets
assets_dir = os.path.join(current_dir, 'assets')
ckd_model_path = os.path.join(assets_dir, 'kidney.sav')

# Debugging output to understand the file structure
st.text(f"Current directory: {current_dir}")
st.text(f"Assets directory: {assets_dir}")
st.text(f"Contents of the current directory: {os.listdir(current_dir)}")
st.text(f"Contents of the assets directory: {os.listdir(assets_dir)}")

# Load the pre-trained model
if os.path.exists(ckd_model_path):
    with open(ckd_model_path, 'rb') as model_file:
        CKD_model = pickle.load(model_file)
else:
    st.error(f"Model file not found: {ckd_model_path}")
    st.stop()

# Sample values from the provided data
sample_values = {
    "age": 48, "bp": 70, "sg": 1.005, "al": 4, "su": 0, "rbc": 'normal', "pc": 'abnormal',
    "pcc": 'present', "ba": 'notpresent', "bgr": 117, "bu": 56, "sc": 3.8, "sod": 111,
    "pot": 2.5, "hemo": 11.2, "pcv": 32, "wc": 6700, "rc": 3.9, "htn": 'yes', "dm": 'no',
    "cad": 'no', "appet": 'poor', "pe": 'yes', "ane": 'yes'
}

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Health Assistant',
                           ['CKD Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity'],
                           default_index=0)

# CKD Prediction Page
if selected == "CKD Prediction":
    st.title('Chronic Kidney Disease (CKD) Prediction using ML')

    # Input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        age = st.number_input('Age', value=sample_values['age'])
    with col2:
        bp = st.number_input('Blood Pressure', value=sample_values['bp'])
    with col3:
        sg = st.number_input('Specific Gravity', value=sample_values['sg'])
    with col4:
        al = st.number_input('Albumin', value=sample_values['al'])
    with col5:
        su = st.number_input('Sugar', value=sample_values['su'])
       
    with col1:
        rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal'], index=0 if sample_values['rbc'] == 'normal' else 1)
    with col2:
        pc = st.selectbox('Pus Cell', ['normal', 'abnormal'], index=0 if sample_values['pc'] == 'normal' else 1)
    with col3:
        pcc = st.selectbox('Pus Cell Clumps', ['notpresent', 'present'], index=0 if sample_values['pcc'] == 'notpresent' else 1)
    with col4:
        ba = st.selectbox('Bacteria', ['notpresent', 'present'], index=0 if sample_values['ba'] == 'notpresent' else 1)
    with col5:
        bgr = st.number_input('Blood Glucose Random', value=sample_values['bgr'])
        
    with col1:
        bu = st.number_input('Blood Urea', value=sample_values['bu'])
    with col2:
        sc = st.number_input('Serum Creatinine', value=sample_values['sc'])
    with col3:
        sod = st.number_input('Sodium', value=sample_values['sod'])
    with col4:
        pot = st.number_input('Potassium', value=sample_values['pot'])
    with col5:
        hemo = st.number_input('Hemoglobin', value=sample_values['hemo'])
         
    with col1:
        pcv = st.number_input('Packed Cell Volume', value=sample_values['pcv'])
    with col2:
        wc = st.number_input('White Blood Cell Count', value=sample_values['wc'])
    with col3:
        rc = st.number_input('Red Blood Cell Count', value=sample_values['rc'])
    with col4:
        htn = st.selectbox('Hypertension', ['yes', 'no'], index=0 if sample_values['htn'] == 'yes' else 1)
    with col5:
        dm = st.selectbox('Diabetes Mellitus', ['yes', 'no'], index=0 if sample_values['dm'] == 'yes' else 1)
    
    with col1:
        cad = st.selectbox('Coronary Artery Disease', ['yes', 'no'], index=0 if sample_values['cad'] == 'yes' else 1)
    with col2:
        appet = st.selectbox('Appetite', ['good', 'poor'], index=0 if sample_values['appet'] == 'good' else 1)
    with col3:
        pe = st.selectbox('Pedal Edema', ['yes', 'no'], index=0 if sample_values['pe'] == 'yes' else 1)
    with col4:
        ane = st.selectbox('Anemia', ['yes', 'no'], index=0 if sample_values['ane'] == 'yes' else 1)

    # Collect user inputs
    user_input = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

    # Convert categorical variables to numerical values
    rbc = 1 if rbc == 'abnormal' else 0
    pc = 1 if pc == 'abnormal' else 0
    pcc = 1 if pcc == 'present' else 0
    ba = 1 if ba == 'present' else 0
    htn = 1 if htn == 'yes' else 0
    dm = 1 if dm == 'yes' else 0
    cad = 1 if cad == 'yes' else 0
    appet = 1 if appet == 'good' else 0
    pe = 1 if pe == 'yes' else 0
    ane = 1 if ane == 'yes' else 0

    # Collect user inputs
    user_input = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

    ckd_prediction = ''
    if st.button('CKD Test Result'):
        try:
            # Ensure all inputs are in the correct format
            user_input = [float(feature) for feature in user_input]

            # Make prediction
            ckd_prediction = CKD_model.predict([user_input])
            if ckd_prediction[0] == 'ckd':
                ckd_diagnosis = 'The person is likely to have Chronic Kidney Disease'
            else:
                ckd_diagnosis = 'The person is not likely to have Chronic Kidney Disease'
            st.success(ckd_diagnosis)
        except ValueError as ve:
            st.error(f'Please enter valid numbers for all fields. ValueError: {ve}')
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
