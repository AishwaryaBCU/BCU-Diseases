import base64
import streamlit as st
import os
import pickle

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
)

# Function to set background image
def set_page_background(image_path):
    @st.cache(suppress_st_warning=True)
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
        # Print the current working directory and contents for debugging
        st.text(f"Current working directory: {os.getcwd()}")
        st.text(f"Contents of the current directory: {os.listdir(os.getcwd())}")

# Set background image path
background_image_path = 'BCU-Diseases/diabetes/background.jpg'

# Set background image
set_page_background(background_image_path)

# Sidebar setup and content
st.sidebar.title("Diabetes Prediction System")
app_mode = st.sidebar.selectbox(
    "Please navigate through the different sections",
    ["Home", "Diabetes Prediction", "Disclaimer"]
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

# Main content
def main():
    if app_mode == "Home":
        st.title("Welcome to Diabetes Prediction System")
        st.write("Select an option from the sidebar to get started.")
    elif app_mode == "Diabetes Prediction":
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
    elif app_mode == "Disclaimer":
        st.title("Disclaimer")
        st.write("""
        The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.
        """)

# Running the app
if __name__ == "__main__":
    # Load the diabetes prediction model
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(working_dir, 'diabetes_model.sav')
    diabetes_model = load_model(model_file_path)
    
    main()
