# Pneumpredict
This repository includes necessary files, codes required to run Pneumpredict and predict the health of lungs from X-rays.

URL to Pneumpredict: https://sabahuddinahmad-pneumpredict-app-v1-ofjwfn.streamlit.app

In recent times, we have seen many successful applications of Machine Learning aided Couputer Vision in health and diagnostics. Interestingly, amongst severel methods in computer vision, image classification stays on the top. This application uses X-ray image classification model that was made using dataset available at Kaggle. The Dataset has been carefully screened by 3 medical experts and is also part of the research article published in the Cell, a reputed journal of the field. 

Jupyter notebook to obtain the image classification model 'Pneumpredict' is included in the main directory of the project.

There are multiple steps involved in the classification process: from loading of the datasets to the predicting of the results of model. App is able to classify the X-ray images into normal/healthy or presence of pneumonia. This was also interesting topic to work on, as the COVID-19 patients develop Pneumonia.

The Sample images can be used for testing the application and are located in the folder called 'sample_images' within the repo. Within the sample_images folder there are subfolders called normal and infected, each containing 50 X-Ray images.
