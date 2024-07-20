import time
import joblib
import pandas as pd
from config import *
import streamlit as st

def prediction_page():
    def convert_to_one_hot(selected_category, all_categories, user_input):
        """Convert a selected category to one-hot encoded vector and append to user_input list."""
        one_hot = [True if category == selected_category else False for category in all_categories]
        user_input.extend(one_hot)

    def predict_alzheimer(input_data):
        """Load the model and predict the Alzheimer's condition."""
        loaded_model = joblib.load('model/alzheimer_model.pkl')
        predictions = loaded_model.predict(input_data)
        return predictions

    st.title("Patient Information")

    # Collect user input
    age = st.number_input("Age", min_value=0, max_value=122, step=1, value=65)
    gender = st.selectbox("Gender", ("Male", "Female"))
    education = st.number_input("Years of Education", min_value=0, value=12)
    ethnicity = st.radio("Ethnicity", ("Hisp/Latino", "Not Hisp/Latino", "Unknown"))
    race_cat = st.radio("Race Category", ("White", "Black", "Asian"))
    apoe_allele_type = st.selectbox("APOE Allele Type", ["APOE4_0", "APOE4_1", "APOE4_2"])
    apoe_genotype = st.selectbox("APOE4 Genotype", ("2,2", "2,3", "2,4", "3,3", "3,4", "4,4"))
    imputed_genotype = st.radio("Imputed Genotype", ("True", "False"))
    mmse = st.number_input("MMSE Score", min_value=0, max_value=30, step=1)

    st.write("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict")

    # Prediction process
    if predict_button:
        # Show progress bar
        loading_bar = st.empty()
        loading_bar.write("Thank you for entering the patient's information.")
        progress_text = "Please wait, we're predicting your clinical condition..."
        my_bar = loading_bar.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Prepare user input for prediction
        user_input = [age, education, mmse]

        # Convert categorical variables to one-hot encoding
        convert_to_one_hot("PTRACCAT_" + race_cat, PTRACCAT_CATEGORIES, user_input)
        convert_to_one_hot("APOE Genotype_" + apoe_genotype, APOE_CATEGORIES, user_input)
        convert_to_one_hot("PTETHCAT_" + ethnicity, PTHETHCAT_CATEGORIES, user_input)
        convert_to_one_hot(apoe_allele_type, APOE4_CATEGORIES, user_input)
        convert_to_one_hot("PTGENDER_" + gender, PTGENDER_CATEGORIES, user_input)
        convert_to_one_hot("imputed_genotype_" + imputed_genotype, IMPUTED_CATEGORIES, user_input)

        # Convert to DataFrame
        data = pd.DataFrame([user_input])
        
        # Predict
        try:
            predicted_condition = predict_alzheimer(data)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return
        finally:
            loading_bar.empty()  # Clear progress bar

        # Display results
        st.write("")
        st.write("### Predicted Clinical Condition:", unsafe_allow_html=True)
        st.write(f"## <b>{ABBREVIATION.get(predicted_condition[0], 'Unknown')}</b> ({predicted_condition[0]})", unsafe_allow_html=True)
        st.write(f"{CONDITION_DESCRIPTION.get(predicted_condition[0], 'No description available.')}", unsafe_allow_html=True)
