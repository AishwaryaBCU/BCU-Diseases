#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model(r"malaria_cell_detection.h5")

#Name of Classes
CLASS_NAMES = ['Parisitized','Healthy']

#Setting Title of App
st.title("Malarial Cell Disease Detection")
st.markdown("Upload an image of the cell")

#Uploading the dog image
cell_image = st.file_uploader("Choose an image...", type="png")
submit = st.button('Predict')
#On predict button click
if submit:


    if cell_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(cell_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (64,64))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,64,64,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title("Cell is "+result)
        
