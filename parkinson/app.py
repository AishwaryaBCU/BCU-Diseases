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

# Function to add custom CSS for background image
def add_background_image():
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

# Adding the background image
add_background_image()

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

    # Input fields with clearer labels
    fo = st.number_input('MDVP: Fo (Hz)', format="%.2f")
    fhi = st.number_input('MDVP: Fhi (Hz)', format="%.2f")
    flo = st.number_input('MDVP: Flo (Hz)', format="%.2f")
    jitter_percent = st.number_input('MDVP: Jitter (%)', format="%.2f")
    jitter_abs = st.number_input('MDVP: Jitter (Abs)', format="%.2f")
    rap = st.number_input('MDVP: RAP', format="%.2f")
    ppq = st.number_input('MDVP: PPQ', format="%.2f")
    ddp = st.number_input('Jitter: DDP', format="%.2f")
    shimmer = st.number_input('MDVP: Shimmer', format="%.2f")
    shimmer_db = st.number_input('MDVP: Shimmer (dB)', format="%.2f")
    apq3 = st.number_input('Shimmer: APQ3', format="%.2f")
    apq5 = st.number_input('Shimmer: APQ5', format="%.2f")
    apq = st.number_input('MDVP: APQ', format="%.2f")
    dda = st.number_input('Shimmer: DDA', format="%.2f")
    nhr = st.number_input('NHR', format="%.2f")
    hnr = st.number_input('HNR', format="%.2f")
    rpde = st.number_input('RPDE', format="%.2f")
    dfa = st.number_input('DFA', format="%.2f")
    spread1 = st.number_input('Spread1', format="%.2f")
    spread2 = st.number_input('Spread2', format="%.2f")
    d2 = st.number_input('D2', format="%.2f")
    ppe = st.number_input('PPE', format="%.2f")

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
