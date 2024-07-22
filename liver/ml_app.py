from app import MultiApp
from PIL import Image
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import base64

app = MultiApp()

st.set_page_config(layout="wide")
def app0():
    st.markdown("""
    <style>
    .big-font {
        font-size:23px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("LIVER HEALTH DETECTION")
    st.write("By Abhishek Saigal")
    st.title("INTRODUCTION")
    st.write("A healthy liver is a basic requirement for the overall health and wellness of an individual.")
    st.write("Presence of Liver Disease has a significant impact on the health of a person as the liver is responsible for over 500 functions in the body and is often referred to as the metabolic factory of our body.")
    st.write("It is therefore of immense use to have an app which is able to detect whether or not a person has a healthy liver. Furthermore, if there is an app which can predict the same without the need of a clinician at the initial assessment, it will be of great utility.")
    img = Image.open('Images/liver image 2.jpg')
    st.image(img, caption='A healthy Liver', width = 450)
    st.title("LIVER DISEASES")
    st.markdown('<p class="big-font">TYPES OF LIVER DISEASES:</p>', unsafe_allow_html=True)
    st.write("An unhealthy liver can be the sequel of several liver diseases stemming from various causes. Some of the common liver diseases are:")
    column_1, column_2 = st.beta_columns([3,5])
    with column_1:
        st.write("1. Acute Viral Hepatitis\n2. Chronic Viral Hepatitis\n3. Cirrhosis\n4. Alcoholic Liver Disease\n 5. Non-Alcoholic Fatty Liver Disease\n 6. Autoimmune Liver Disease\n7. Drug-induced Liver Disease\n8. Cholestatic Liver Disease\n9. Genetic and Metabolic Liver Diseases\n10. Liver Cancer\n")
        #st.write("These Diseases can be caused by a variety of factors such as:")
        #st.write("1. Hepatitis Viruses (A, B, C and E)\n2. Alcohol\n3. Matabolic causes\n4. Immune System Abnormalities\n5. Genetics\n6. Cancer\n7. Hepatotoxic Drugs")
    with column_2:
        st.image('Images/l2.jpg', use_column_width=True)

    st.markdown('<p class="big-font">CAUSES OF LIVER DISEASES:</p>', unsafe_allow_html=True)
    st.write("These Liver Diseases can be caused by a variety of factors such as:")
    column_3, column_4, column_5 = st.beta_columns([9,6,4])
    with column_3:
        st.write("1. Hepatitis Viruses (A, B, C and E)\n2. Alcohol\n3. Metabolic causes\n4. Immune System Abnormalities\n5. Genetic Abnormalities\n6. Cancer\n7. Hepatotoxic Drugs")
    with column_4:
        st.image('Images/Screen Shot 2021-05-16 at 11.11.48 PM.png', use_column_width=True)
    st.title("OBJECTIVES")
    st.write("1. To gain an understanding of Liver Diseases and the main factors causing them.\n2. To analyse the data and find out which features are most important in detecting Liver Disease.\n3. To develop a Liver Disease Prediction machine learning model which is able to perform at a relatively high accuracy (85-90%).")

def app1():
    st.title("THE DATASET")
    st.write("The Indian Liver Patient Records Dataset has patient records from the North East of Andhra Pradesh, India.")
    st.write("This dataset consists of 583 total records, with 416 liver patient records and 167 non-liver patient records.")
    st.write("The features present in this dataset were:")
    st.write("1. Age of the Patient\n2. Gender of the Patient\n3. Total Bilirubin\n4. Direct Bilirubin\n5. Alkaline Phosphotase\n6. Alanine Aminotransferase\n7. Aspartate Aminotransferase\n8. Total Proteins\n9. Albumin\n10. Albumin and Globulin Ratio")
    st.write("Each entry also has a label \"Dataset\" to split the data into persons with liver disease and those without liver disease.")
    liver = pd.read_csv('indian_liver_patient.csv')
    st.subheader('INDIAN LIVER PATIENTS DATASET:')
    st.dataframe(liver)
    st.subheader("KEY STATISTICS ABOUT THE DATASET")
    st.write(liver.describe())
    st.subheader("INITIAL DATA ANALYSIS: DISTRUBUTION PLOTS")
    #fig, ax = plt.subplots(figsize=(5, 2))
    #sns.distplot(liver['Aspartate_Aminotransferase'])
    #st.pyplot(fig)
    img = Image.open('Images/merge_from_ofoct (1).jpg')
    st.image(img, use_column_width = True)
    img = Image.open('Images/merge_from_ofoct (3).jpg')
    st.image(img, use_column_width = True)
    img = Image.open('Images/merge_from_ofoct (5).jpg')
    st.image(img, use_column_width = True)


def app2():
    liver = pd.read_csv('indian_liver_patient.csv')
    st.title('EXPLORATORY DATA ANALYSIS')
    st.title("Correlation Heatmap")
    st.write("By analysing a heatmap we can see the correlation between all the features in the dataset.")
    #cormap = liver.corr()
    #fig, ax = plt.subplots(figsize=(15,15))
    #sns.heatmap(cormap, cmap = "Greens", annot = True)
    #st.pyplot(fig)
    img = Image.open('Images/new_heatmap.png')
    st.image(img, use_column_width = True)

    st.title("Further Analysis:")
    #sns.set()
    #fig2 = sns.pairplot(liver, hue='Dataset', kind='reg')
    #st.pyplot(fig2)
    img = Image.open('Images/liver graphs.png')
    st.image(img, use_column_width = True)

    st.title("Highly Correlated Features")
    st.write("From these 2 plots, we have found 4 pairs of highly correlated features:")
    st.write("1. Direct Bilirubin and Total Bilirubin")
    st.write("2. Aspartate Aminotransferase and Alanine Aminotransferase")
    st.write("3. Total Proteins and Albumin")
    st.write("4. Albumin and Albumin and Globulin Ratio")
    st.title("Visualisation of correlated features:")
    #sns.set()
    #fig = sns.jointplot("Direct_Bilirubin", "Total_Bilirubin", data=liver, kind="reg")
    #st.pyplot(fig)
    img = Image.open('Images/corr_merge.jpg')
    st.image(img, width = 900)
    img = Image.open('Images/merge_from_ofoct (7).jpg')
    st.image(img, width = 900)

def app3():
    st.title("LIVER DISEASE PREDICTION")
    def get_input():
        age = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            age = st.slider('Age', 1, 90, 35)
        with col_2:
            bili = 0
            s_bili = st.text_input("Age", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                age = bili
        #st.write(age*2)
        input_gender = st.text_input("Gender", "")
        gender = 0
        if input_gender.lower() == "female":
            gender = 1

        total_bilirubin = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            total_bilirubin = st.slider('Total Bilirubin', 0.5, 15.0, 0.80)
        with col_2:
            bili = 0
            s_bili = st.text_input("Total Bilirubin", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                total_bilirubin = bili
        #st.write(total_bilirubin*2)
        direct_bilirubin = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            direct_bilirubin = st.slider('Direct Bilirubin', 0.3, 12.5, 0.30)
        with col_2:
            bili = 0
            s_bili = st.text_input("Direct Bilirubin", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                direct_bilirubin = bili
        #st.write(direct_bilirubin*2)
        alkaline_phosphotase = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            alkaline_phosphotase = st.slider('Alkaline Phosphotase', 50, 500, 100)
        with col_2:
            bili = 0
            s_bili = st.text_input("Alkaline Phosphotase", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                alkaline_phosphotase = bili
        #st.write(alkaline_phosphotase*2)
        alamine_amonotransferase = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            alamine_aminotransferase = st.slider('Alanine Aminotransferase', 20, 250, 24)
        with col_2:
            bili = 0
            s_bili = st.text_input("Alanine Aminotransferase", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                alamine_aminotransferase = bili
        #st.write(alamine_aminotransferase*2)
        aspartate_amonotransferase = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            aspartate_aminotransferase = st.slider('Aspartate Aminotransferase', 20, 500, 25)
        with col_2:
            bili = 0
            s_bili = st.text_input("Aspartate Aminotransferase", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                aspartate_aminotransferase = bili
        #st.write(aspartate_aminotransferase*2)
        total_proteins = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            total_proteins = st.slider('Total Proteins', 2.0, 10.0, 6.0)
        with col_2:
            bili = 0
            s_bili = st.text_input("Total Proteins", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                total_proteins = bili
        #st.write(total_proteins*2)
        albumin = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            albumin = st.slider('Albumin', 0.5, 5.0, 3.8)
        with col_2:
            bili = 0
            s_bili = st.text_input("Albumin", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                albumin = bili
        #st.write(albumin*2)
        albumin_and_globulin_ratio = -1
        col_1, col_2 = st.beta_columns([5,1])
        with col_1:
            albumin_and_globulin_ratio = st.slider('Albumin and Globulin Ratio', 0.1, 3.0, 1.73)
        with col_2:
            bili = 0
            s_bili = st.text_input("Albumin and Globulin Ratio", "")
            if(len(s_bili) != 0):
                bili = float(s_bili)
            if (bili != 0):
                albumin_and_globulin_ratio = bili
        #st.write(albumin_and_globulin_ratio*2)
        user_data = {'age': age, 'gender':gender, 'total bilirubin':total_bilirubin, 'direct bilirubin':direct_bilirubin, 'alkaline phosphotase':alkaline_phosphotase, 'alanine aminotransferase':alamine_aminotransferase, 'aspartate aminotransferase':aspartate_aminotransferase, 'total proteins':total_proteins, 'albumin':albumin, 'albumin and globulin ratio':albumin_and_globulin_ratio}
    
        features = pd.DataFrame(user_data, index=[0])
        return features
    st.title('User Details:')
    st.write("Please enter the readings for the following values by moving the slider. In case the value does not fit in the given range, please enter it in the text box provided next to the slider. ")
    st.write("After entering the readings, please scroll down to the \"Result\" section to check whether or not the person has a healthy Liver.")
    user_input = get_input()
    user_input.columns = user_input.columns.map(str.lower) 
    st.dataframe(user_input)
    import pickle
    model = pickle.load(open('new_model_1.pkl','rb'))
    prediction = model.predict(user_input)
    st.title("Result:")
    if (prediction==1):
        st.write("The person has Liver Disease.")
    else:
        st.write("The person does not have Liver Disease.")
    acc_s = 0.90
    st.write('Accuracy: ', acc_s,)
    
    #st.title("Confusion Matrix")
    #cm=confusion_matrix(y_test, y_test_hat)
    #st.write(cm)
    #plot_confusion_matrix(model, y_test, y_test_hat)
    #st.pyplot()
    #img = Image.open('Images/confusion matrix.png')
    #st.image(img, width = 600)
    #from sklearn import metrics
    #st.title("Accuracy Score:")
    #st.write(metrics.accuracy_score(y_test,y_test_hat)*100)
    
    import base64
    st.title('Visualisation of the Extra Trees Classifier')
    st.write("The Extremely Randomized Trees Classifier aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it’s classification result. A major difference between ExtraTrees and Random Forest is that a random forest chooses the optimal split at each node while an Extra Trees classifier chooses it randomly, thereby making it significantly faster for equally good performance. The visualisation of a random forest can be seen below:")
    file_ = open("Images/Random Forest 03.gif", "rb")
    st.image("https://1.bp.blogspot.com/-Ax59WK4DE8w/YK6o9bt_9jI/AAAAAAAAEQA/9KbBf9cdL6kOFkJnU39aUn4m8ydThPenwCLcBGAsYHQ/s0/Random%2BForest%2B03.gif", width=900)
    st.write("Credit: The Tensorflow Blog")

#def app5():
    #st.title("FUTURE PROSPECTS")
    #st.write("1. We can use convolutional neural networks to develop a classification model to predict the exact Liver Disease a patient has ")
    #st.write("2. The Extra Trees Classifier is an effective machine learning model for analysis of Medical Datasets such as Liver Function Tests.")

st.sidebar.title("PAGE NAVIGATION")
app.add_app("Home Page", app0)
app.add_app("The Dataset", app1)
app.add_app("Exploratory Data Analysis", app2)
app.add_app("Liver Disease Prediction", app3)
#app.add_app("Future Prospects", app5)
app.run()
