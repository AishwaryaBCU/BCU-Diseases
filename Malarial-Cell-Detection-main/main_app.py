import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import os


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
