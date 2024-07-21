import pandas as pd
import streamlit as st
import tensorflow as tf
import time
from PIL import Image, ImageOps
import numpy as np
import webbrowser
import os

st.set_page_config(layout="wide",page_icon="🧑‍⚕️", page_title="Pneumonia prediction    ")

options = st.sidebar.radio('PNEUMPREDICT MENU', options=['🏠Home', '🏥About Pneumonia', '🤖Application', '⚠️Disclaimer', '🔖Resources', '👨🏻‍💻About Project'])

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def Ho():
    st.title(":red[_Pneumpredict_]")
    st.write(":grey[Web App for PNEUMonia PREDICTion using X-ray image classifications]")

    home_img_path = os.path.join(current_dir, 'web_img', 'home.jpg')
    if not os.path.exists(home_img_path):
        st.error(f"File not found: {home_img_path}")
    else:
        home_img = Image.open(home_img_path)
        st.image(home_img, width=800)

def Ab():
    st.header(':red[What is Pneumonia?]')
    video = "https://upload.wikimedia.org/wikipedia/commons/d/d5/En.Wikipedia-VideoWiki-Pneumonia.webm"
    st.video(video, format="video/mp4", start_time=0)
    st.write("Source and further reading available at https://en.wikipedia.org/wiki/Pneumonia")

def Ap():
    @st.cache_data
    def load_model():
        model_path = os.path.join(current_dir, 'xray_model_80-20.h5')
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        model = tf.keras.models.load_model(model_path)
        return model

    with st.spinner('Please wait, while the model is being loaded..'):
        model = load_model()
        if model is None:
            st.error("Failed to load the model. Please check the file path and try again.")
            return

    def main():
        st.header(":red[Pneumonia prediction using _Pneumpredict_]")

    if __name__ == '__main__':
        main()

    file = st.file_uploader(" ", accept_multiple_files=False, help="Only one file at a time. The image should be of good quality")

    if file is None:
        st.subheader("Please upload an X-ray image using the browse button :point_up:")
        st.write("Sample images can be found [here](https://github.com/sabahuddinahmad/Pneumpredict/tree/main/sample_images)!")
        compared_img_path = os.path.join(current_dir, 'web_img', 'compared.JPG')
        if not os.path.exists(compared_img_path):
            st.error(f"File not found: {compared_img_path}")
        else:
            image1 = Image.open(compared_img_path)
            st.image(image1, use_column_width=True)
    else:
        st.subheader("Thank you for uploading X-ray image!")
        with st.spinner('_Pneumpredict_ is now processing your image.......'):
            try:
                img = Image.open(file)
                img = ImageOps.grayscale(img)  # Convert image to grayscale
                img = img.resize((500, 500))  # Resize image to 500x500
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=-1)  # Add the channel dimension
                img_array = np.expand_dims(img_array, axis=0)  # Create a batch

                # Debugging information
                st.write(f"Image shape after preprocessing: {img_array.shape}")

                predictions = model.predict(img_array)
                score = tf.sigmoid(predictions)

                time.sleep(2)
                st.success('Prediction is complete!')
                st.subheader(
                    f"Uploaded X-ray image looks like this :point_down: and most likely belongs to {'Infected lungs' if np.max(score) > 0.5 else 'Normal lungs'}!"
                )
                st.image(img, width=400)
                st.subheader("Thank you for using _Pneumpredict_")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

def Di():
    disclaimer_img_path = os.path.join(current_dir, 'web_img', 'disclaimer.JPG')
    if not os.path.exists(disclaimer_img_path):
        st.error(f"File not found: {disclaimer_img_path}")
    else:
        image2 = Image.open(disclaimer_img_path)
        st.image(image2, use_column_width=True)
    st.subheader('This App does not substitute a healthcare professional!')
    st.header('')
    st.write('1. Accuracy of prediction depends on the datasets which were used for training the model within this App, and also depends on the quality of image provided.')
    st.write('2. Do not use prediction results from this App to diagnose or treat any medical or health condition.')
    st.write('3. App cannot classify underlying medical reasons that corresponds to the infections, for example: bacterial, viral, smoking, etc.')
    st.write('4. Healthcare professional will do blood tests and other physical examinations to identify root cause of the infections.')
    st.write('5. Uplodaded X-ray image is not retained by _Pneumpredict_.')

def Ci():
    st.header(':red[Dataset availibility & recommended resources:]')
    st.subheader('')
    st.write("1. Dataset used for this project is available as [Chest X-ray Images at Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).")
    st.write("2. Above dataset is part of a [publication](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5), _Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning_.")
    st.write("3. Inspiration for TensorFlow implementation in image classification on above dataset was from a [Notebook on Kaggle by Amy Jang](https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays).")
    st.write("4. To implement TensorFlow in image classification, there is an amazing [tutorial](https://www.tensorflow.org/tutorials/images/classification).")

def Me():
    st.header(':red[About Project:]')
    st.subheader('')
    st.write('Greetings! I am Aishwarya U ,Student at Bengaluru City University, This project deals with finding whether the person suffering with Pneumonia or not.')
   
    st.write('Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.')
    st.write('The machine learning model used in this project achieves an accuracy of approximately 92% on the test dataset. This means that the model is able to correctly predict pneumonia 92% of the time based on the given health parameters.')
    st.write('This pneumonia prediction tool is developed using machine learning techniques to predict the likelihood of pneumonia in individuals based on various health parameters. The model is trained on a dataset that includes parameters.')

if options == '🏠Home':
    Ho()
elif options == '🏥About Pneumonia':
    Ab()
elif options == '🤖Application':
    Ap()
elif options == '⚠️Disclaimer':
    Di()
elif options == '🔖Resources':
    Ci()
elif options == '👨🏻‍💻About Project':
    Me() 
