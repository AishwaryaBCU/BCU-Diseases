import pandas as pd
import streamlit as st
import tensorflow as tf
import time
from PIL import Image, ImageOps
import numpy as np
import webbrowser
import os

st.set_page_config(layout="wide")

options = st.sidebar.radio('PNEUMPREDICT MENU', options=['🏠Home', '🏥About Pneumonia', '🤖Application', '⚠️Disclaimer', '🔖Resources', '👨🏻‍💻About me'])

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def Ho():
    st.title(":red[_Pneumpredict_]")
    st.write(":grey[Web App for PNEUMonia PREDICTion using X-ray image classifications]")

    home_img_path = os.path.join(current_dir, 'web_img', 'home.jpg')
    home_img = Image.open(home_img_path)
    st.image(home_img, width=800)

def Ab():
    st.header(':red[What is Pneumonia?]')
    video = "https://upload.wikimedia.org/wikipedia/commons/d/d5/En.Wikipedia-VideoWiki-Pneumonia.webm"
    st.video(video, format="video/mp4", start_time=0)
    st.write("Source and further reading available at https://en.wikipedia.org/wiki/Pneumonia")

def Ap():
    @st.cache(allow_output_mutation=True)
    def load_model():
        model_path = os.path.join(current_dir, 'xray_model_80-20.h5')
        model = tf.keras.models.load_model(model_path)
        return model

    with st.spinner('Please wait, while the model is being loaded..'):
        model = load_model()

    def main():
        st.header(":red[Pneumonia prediction using _Pneumpredict_]")

    if __name__ == '__main__':
        main()

    file = st.file_uploader(" ", accept_multiple_files=False, help="Only one file at a time. The image should be of good quality")

    if file is None:
        st.subheader("Please upload an X-ray image using the browse button :point_up:")
        st.write("Sample images can be found [here](https://github.com/sabahuddinahmad/Pneumpredict/tree/main/sample_images)!")
        compared_img_path = os.path.join(current_dir, 'web_img', 'compared.JPG')
        image1 = Image.open(compared_img_path)
        st.image(image1, use_column_width=True)
    else:
        st.subheader("Thank you for uploading X-ray image!")
        with st.spinner('_Pneumpredict_ is now processing your image.......'):
            img = Image.open(file)
            img = img.resize((180, 180))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.sigmoid(predictions)

            time.sleep(2)
            st.success('Prediction is complete!')
            st.subheader(
                f"Uploaded X-ray image looks like this :point_down: and most likely belongs to {'Infected lungs' if np.max(score) > 0.5 else 'Normal lungs'}!"
            )
            st.image(img, width=400)
            st.subheader("Thank you for using _Pneumpredict_")

def Di():
    disclaimer_img_path = os.path.join(current_dir, 'web_img', 'disclaimer.JPG')
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
    st.header(':red[About myself:]')
    st.subheader('')
    st.write('Greetings! I am Sabah, a computational biochemist passionate about decoding the mysteries of life through molecular modeling & data science. With a doctorate in Computational Biochemistry and a background in Bioinformatics, I have contributed to significant research in biomedicine.')
    st.write('From unraveling the dynamics of membrane proteins to pioneering novel drug discovery methods, my work has been [published](https://scholar.google.com/citations?user=fvF831wAAAAJ&hl=en) in leading journals and has taken me around the globe, collaborating with top researchers.')
    st.write('Driven by curiosity, I recently immersed myself in data science, completing a 4 months long rigorous bootcamp at WBS Coding School in Berlin. There for the final project (3 weeks long), I developed _Pneumpredict_, a tool using machine learning to classify lung X-rays and identify lung infections.')
    st.write('Connect with me on [LinkedIn](https://www.linkedin.com/in/sabahuddinahmad) or follow me on [Twitter](https://twitter.com/sabahahmad_IN) to stay updated on my latest projects and professional endeavors.')
    st.write('Let us connect, collaborate, and explore new horizons together!')

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
elif options == '👨🏻‍💻About me':
    Me()
