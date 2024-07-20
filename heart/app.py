import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


def load_models():
    models = {
        'Logistic Regression': pickle.load(open('heart_disease_model.sav', 'rb')),
        
    }
    return models


def predict_risk(models, selected_model, features):
    model = models[selected_model]
    prediction = model.predict(features)
    return prediction[0]


def show_user_inputs():
    st.title("Heart Disease Risk Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age ", step=1, min_value=1)
        trestbps = st.number_input("Resting Blood Pressure", step=1, min_value=1)
        restecg = st.selectbox("Resting Electrocardiographic Results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable ", [0, 1, 2])
        oldpeak = st.number_input("ST Depression Induced by Exercise")
        thal = st.selectbox("Thal", [1, 2, 3])

    with col2:
        sex = st.selectbox("Sex (1 = male, 0 = female)", [0, 1])
        chol = st.number_input("Serum Cholestoral in mg/dl", step=1, min_value=1)
        thalach = st.number_input("Maximum Heart Rate Achieved", step=1, min_value=1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])

    with col3:
        cp = st.selectbox("Chest Pain Type  (0= typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)", [0,1, 2, 3])
        fbs = st.selectbox("Fasting blood sugar level (1 = >120 mg/dL, 0 = <=120 mg/dL).", [0, 1])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        ca = st.selectbox("Major Vessels Colored by Flourosopy", [0, 1, 2, 3])

    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]


def main():
    st.set_page_config(page_title="Disease Prediction", page_icon="+", layout="wide")
    with st.sidebar:
        models_list = [
                       'Logistic Regression'
                      ]

        selected_model = option_menu('Heart Disease Prediction', models_list, icons=['heart'], default_index=0)
    models = load_models()

    if selected_model == 'Working models':
        features = show_user_inputs()
        if st.button("Predict"):
            results = {}
            for model_name in models:
                result = predict_risk(models, model_name, [features])
                results[model_name] = result

            results_df = pd.DataFrame(results.items(), columns=['Model', 'Result'])
            results_df['Result'] = results_df['Result'].astype(str)

           
            for model_name, result in results.items():
                st.write(model_name + ":")
                if result == 1:
                    st.success("The person has a risk of heart disease.")
                else:
                    st.success("The person does not have a risk of heart disease.")
    else:
        if selected_model in models:
            features = show_user_inputs()

            if st.button("Predict"):
                result = predict_risk(models, selected_model, [features])
                st.write(selected_model + ":")
                if result == 0:
                    st.success("The person has a risk of heart disease.")
                else:
                    st.success("The person does not have a risk of heart disease.")
        else:
            st.write("Model unavailable")


if __name__ == "__main__":
    main()
