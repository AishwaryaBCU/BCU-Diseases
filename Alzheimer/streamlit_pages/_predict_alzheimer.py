import time
import joblib
import pandas as pd
from config import *
import streamlit as st

def prediction_page():
    def convert_to_one_hot(selected_category, all_categories):
        one_hot = [True if category == selected_category else False for category in all_categories]
        return one_hot

    def predict_alzheimer(input_data):
        try:
            # Load the model using the path from config
            loaded_model = joblib.load(MODEL_PATH)
            predictions = loaded_model.predict(input_data)
            return predictions
        except FileNotFoundError:
            st.error(f"Model file not found: {MODEL_PATH}")
            return None
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return None

    st.title("Patient Information")

    age = st.number_input("Age", min_value=0, max_value=122, step=1, value=65)
    gender = st.selectbox("Gender", ("Male", "Female"))
    education = st.number_input("Years of Education", min_value=0, value=12)

    st.write("<br>", unsafe_allow_html=True)

    st.header("Demographics")
    ethnicity = st.radio("Ethnicity", ("Hisp/Latino", "Not Hisp/Latino", "Unknown"))
    race_cat = st.radio("Race Category", ("White", "Black", "Asian"))

    st.write("<br>", unsafe_allow_html=True)

    st.header("Genetic Information")
    apoe_allele_type = st.selectbox("APOE Allele Type", ["APOE4_0", "APOE4_1", "APOE4_2"])
    apoe_genotype = st.selectbox("APOE4 Genotype", ("2,2", "2,3", "2,4", "3,3", "3,4", "4,4"))
    imputed_genotype = st.radio("Imputed Genotype", ("True", "False"))

    st.header("Cognitive Assessment")
    mmse = st.number_input("MMSE Score", min_value=0, max_value=30, step=1)

    st.write("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict")
    st.write("")

    loading_bar = st.empty()
    if predict_button:
        user_input = [age, education, mmse]

        user_input.extend(convert_to_one_hot("PTRACCAT_" + race_cat, PTRACCAT_CATEGORIES))
        user_input.extend(convert_to_one_hot("APOE Genotype_" + apoe_genotype, APOE_CATEGORIES))
        user_input.extend(convert_to_one_hot("PTETHCAT_" + ethnicity, PTHETHCAT_CATEGORIES))
        user_input.extend(convert_to_one_hot(apoe_allele_type, APOE4_CATEGORIES))
        user_input.extend(convert_to_one_hot("PTGENDER_" + gender, PTGENDER_CATEGORIES))
        user_input.extend(convert_to_one_hot("imputed_genotype_" + imputed_genotype, IMPUTED_CATEGORIES))

        data = pd.DataFrame([user_input])

        # Show loading bar while predicting
        with loading_bar.container():
            st.write("Thank you for entering the patient's information.")
            progress_text = "Please wait, we're predicting your clinical condition..."
            my_bar = loading_bar.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1, text=progress_text)

        predicted_condition = predict_alzheimer(data)
        
        if predicted_condition is not None:
            loading_bar.empty()
            st.write("")
            st.write("")
            st.write("### Predicted Clinical Condition:", unsafe_allow_html=True)
            st.write(f"## <b>{ABBREVIATION.get(predicted_condition[0], 'Unknown')}</b> ({predicted_condition[0]})", unsafe_allow_html=True)
            st.write(f"{CONDITION_DESCRIPTION.get(predicted_condition[0], 'Description not available.')}", unsafe_allow_html=True)
