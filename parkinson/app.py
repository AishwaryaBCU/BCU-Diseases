import os
import pickle
import base64
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Parkinson's Disease Prediction",
                   layout="wide",
                   page_icon="🧠")  # Page icon updated to brain emoji

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved model
parkinsons_model = pickle.load(open(os.path.join(working_dir, 'parkinsons_model.sav'), 'rb'))

# Function to add custom CSS for background image and compact input fields
def add_custom_css():
    try:
        background_image = get_base64_of_file(os.path.join(working_dir, 'bg.webp'))
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/webp;base64,{background_image}");
                background-size: cover;
                background-position: center;
            }}
            .stNumberInput input {{
                width: 100px;
                height: 30px;
                padding: 5px;
                font-size: 14px;
            }}
            .stTextInput input {{
                width: 100px;
                height: 30px;
                padding: 5px;
                font-size: 14px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Background image file not found.")

# Function to get base64 encoding of the image file
def get_base64_of_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Adding custom CSS
add_custom_css()

# Navigation
st.title("Parkinson's Disease Prediction")
page = st.selectbox("Select a page:", ["Home", "Parkinson's Prediction", "Disclaimer"])

if page == "Home":
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

elif page == "Parkinson's Prediction":
    st.write("## Parkinson's Disease Prediction using ML")

    # Input fields with compact size
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.number_input('MDVP: Fo (Hz)', format="%.2f", key='fo')
    with col2:
        fhi = st.number_input('MDVP: Fhi (Hz)', format="%.2f", key='fhi')
    with col3:
        flo = st.number_input('MDVP: Flo (Hz)', format="%.2f", key='flo')
    with col4:
        jitter_percent = st.number_input('Jitter (%)', format="%.2f", key='jitter_percent')
    with col5:
        jitter_abs = st.number_input('Jitter (Abs)', format="%.2f", key='jitter_abs')
    with col1:
        rap = st.number_input('RAP', format="%.2f", key='rap')
    with col2:
        ppq = st.number_input('PPQ', format="%.2f", key='ppq')
    with col3:
        ddp = st.number_input('DDP', format="%.2f", key='ddp')
    with col4:
        shimmer = st.number_input('Shimmer', format="%.2f", key='shimmer')
    with col5:
        shimmer_db = st.number_input('Shimmer (dB)', format="%.2f", key='shimmer_db')
    with col1:
        apq3 = st.number_input('APQ3', format="%.2f", key='apq3')
    with col2:
        apq5 = st.number_input('APQ5', format="%.2f", key='apq5')
    with col3:
        apq = st.number_input('APQ', format="%.2f", key='apq')
    with col4:
        dda = st.number_input('DDA', format="%.2f", key='dda')
    with col5:
        nhr = st.number_input('NHR', format="%.2f", key='nhr')
    with col1:
        hnr = st.number_input('HNR', format="%.2f", key='hnr')
    with col2:
        rpde = st.number_input('RPDE', format="%.2f", key='rpde')
    with col3:
        dfa = st.number_input('DFA', format="%.2f", key='dfa')
    with col4:
        spread1 = st.number_input('Spread1', format="%.2f", key='spread1')
    with col5:
        spread2 = st.number_input('Spread2', format="%.2f", key='spread2')
    with col1:
        d2 = st.number_input('D2', format="%.2f", key='d2')
    with col2:
        ppe = st.number_input('PPE', format="%.2f", key='ppe')

    if st.button("Get Prediction"):
        try:
            user_input = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                          apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
            prediction = parkinsons_model.predict([user_input])
            if prediction[0] == 1:
                st.success("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif page == "Disclaimer":
    st.write("## Disclaimer")
    st.write("""
    The information provided by this application is for educational purposes only. The predictions generated by this tool are based on machine learning models and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or accessed through this application.
    """)

# Running the app
if __name__ == '__main__':
    st.success('Welcome to the Health Assistant application!')
