import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧠")  # Updated page icon to brain emoji

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
parkinsons_model = pickle.load(open(os.path.join(working_dir, 'parkinsons_model.sav'), 'rb'))

# Function to add custom CSS for background image
def add_background_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{get_base64_of_file('bg.webp')}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to get base64 encoding of the image file
def get_base64_of_file(file_path):
    import base64
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Adding the background image
add_background_image()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Machine Learning Diseases Prediction System',
                           [
                               'Home',
                               'Parkinsons Prediction',
                               'Disclaimer'
                           ],
                           menu_icon='hospital-fill',
                           icons=['house', 'person', 'exclamation-triangle'],
                           default_index=0)

# Home Page
if selected == "Home":
    st.title("Health Assistant")
    st.write("## About Parkinson's Disease")
    st.write("""
    Parkinson's disease is a progressive nervous system disorder that affects movement. Symptoms start gradually, sometimes starting with a barely noticeable tremor in just one hand. Tremors are common, but the disorder also commonly causes stiffness or slowing of movement.
    """)
    st.write("## Machine Learning Project")
    st.write("""
    This Parkinson's disease prediction tool is developed using machine learning techniques to predict the likelihood of Parkinson's disease in individuals based on various health parameters. The model is trained on the Parkinson's Disease Dataset, which includes parameters such as:

    - MDVP: Fo(Hz) - Average vocal fundamental frequency
    - MDVP: Fhi(Hz) - Maximum vocal fundamental frequency
    - MDVP: Flo(Hz) - Minimum vocal fundamental frequency
    - MDVP: Jitter(%) - Variation in fundamental frequency
    - MDVP: Shimmer - Variation in amplitude
    - HNR - Harmonics-to-noise ratio
    - RPDE - Recurrence period density entropy
    - DFA - Detrended fluctuation analysis
    - Spread1 - Nonlinear measures of fundamental frequency variation
    - Spread2 - Nonlinear measures of fundamental frequency variation
    - D2 - Correlation dimension
    - PPE - Pitch period entropy

    The machine learning model used in this project achieves an accuracy of approximately 90% on the test dataset. This means that the model is able to correctly predict Parkinson's disease 90% of the time based on the given health parameters.
    """)
    st.write("[Go to Online Predictor](#)")

# Parkinson's Prediction Page
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                          float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                          float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                          float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error('Please enter valid numbers for all fields.')

# Disclaimer Page
elif selected == "Disclaimer":
    st.title("Disclaimer")
    st.write("""
    The information provided by this application is for educational purposes only. The predictions generated by this tool are based on machine learning models and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or accessed through this application.
    """)

# Running the app
if __name__ == '__main__':
    st.success('Welcome to the Health Assistant application!')
