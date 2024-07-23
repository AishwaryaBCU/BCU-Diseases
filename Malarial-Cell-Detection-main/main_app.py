import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import os


st.set_page_config(
    page_title="Chronic Kidney Disease Predictor",
    page_icon="🦠",
    layout="wide"
)

# Function to load the model with error handling
def load_model_safely(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except IOError as e:
        st.error(f"Error loading model: {e}")
        return None

# Path to the model file within the subdirectory
subdirectory = 'Malarial-Cell-Detection-main'
model_filename = 'malaria_cell_detection.h5'
model_path = os.path.join(subdirectory, model_filename)

# Main App
def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Home", "Disclaimer", "Predict"])

    if option == "Home":
        show_home_page()
    elif option == "Disclaimer":
        show_disclaimer_page()
    elif option == "Predict":
        show_prediction_page()

def show_home_page():
    st.title("Welcome to the Malarial Cell Detection App")
    st.markdown("""
    This application uses a Convolutional Neural Network (CNN) to detect whether a cell is infected with malaria or not. 

    **How to use:**
    1. Go to the "Predict" page.
    2. Upload an image of a cell.
    3. Click the "Predict" button to get the results.

    This model has been trained on a dataset of cell images and can classify them as either 'Parasitized' or 'Healthy'.
    """)

def show_disclaimer_page():
    st.title("Disclaimer")
    st.markdown("""
    **Disclaimer:**
    
    The results provided by this application are based on a machine learning model and are for informational purposes only. 

    While we strive for accuracy, the model's predictions should not be used as a substitute for professional medical advice or diagnosis. Always consult with a healthcare professional for accurate medical advice.

    The developers of this application are not responsible for any decisions made based on the results provided by this tool.
    """)

def show_prediction_page():
    # Loading the Model
    model = load_model_safely(model_path)

    # Name of Classes
    CLASS_NAMES = ['Parasitized', 'Healthy']

    # Setting Title of App
    st.title("Malarial Cell Disease Detection")
    st.markdown("Upload an image of the cell")

    # Uploading the cell image
    cell_image = st.file_uploader("Choose an image...", type="png")
    submit = st.button('Predict')

    # On predict button click
    if submit:
        if cell_image is not None:
            try:
                # Convert the file to an OpenCV image
                file_bytes = np.asarray(bytearray(cell_image.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)

                # Displaying the image
                st.image(opencv_image, channels="BGR")
                st.write(opencv_image.shape)

                # Resizing the image
                opencv_image = cv2.resize(opencv_image, (64, 64))

                # Convert image to 4 dimensions
                opencv_image = np.expand_dims(opencv_image, axis=0)

                # Make prediction
                Y_pred = model.predict(opencv_image)
                result = CLASS_NAMES[np.argmax(Y_pred)]
                st.title(f"Cell is {result}")
            except Exception as e:
                st.error(f"Error processing the image: {e}")
        else:
            st.error("Please upload an image.")
    else:
        st.info("Click 'Predict' to analyze the image.")

if __name__ == "__main__":
    main()
