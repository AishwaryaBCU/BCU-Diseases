import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import os

# Function to load the model with error handling
def load_model_safely(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except IOError as e:
        st.error(f"Error loading model: {e}")
        return None

# Verify the current working directory
current_working_directory = os.getcwd()
st.text(f"Current working directory: {current_working_directory}")

# List files in the current working directory
files_in_directory = os.listdir(current_working_directory)
st.text(f"Files in the directory: {files_in_directory}")

# Path to the model file
model_filename = 'malaria_cell_detection.h5'
model_path = os.path.join(current_working_directory, model_filename)

# Verify the absolute path to the model file
st.text(f"Model path: {model_path}")

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
