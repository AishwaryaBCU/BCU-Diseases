import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os

st.set_page_config(
    page_title="Chronic Kidney Disease Predictor",
    page_icon="👨‍⚕️",
    layout="wide"
)

st.title('👨‍⚕️ Chronic Kidney Disease Predictor')

st.markdown("Chronic Kidney Disease (CKD) is a condition where your kidneys don't work as well as they should for a long time. It can make you feel tired, swollen, or have trouble thinking clearly. This web app predicts if a patient has **Chronic Kidney Disease (CKD)** based on the patient's data.")

total_features = 24

if 'omit_feat' not in st.session_state:
    st.session_state.omit_feat = []
    st.session_state.omit_feat_mat = np.zeros(total_features, dtype=bool)

# Get the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load column info
column_info_path = os.path.join(base_dir, 'assets', 'column_info.json')
with open(column_info_path, 'r') as file:
    column_info = json.load(file)

labels = column_info['full']
X = pd.DataFrame(np.empty((1, total_features)), columns=labels)

def disable_widgets():
    st.session_state.omit_feat_mat = np.zeros(total_features, dtype=bool)
    indices = [labels.index(item) for item in st.session_state.omit_feat if item in labels]
    st.session_state.omit_feat_mat[indices] = True

st.header("Input the Patient's Data")
omit_feat = st.multiselect("Select the features you don't know", labels, 
                            placeholder="Omitted Features e.g. Potassium (I don't know the potassium level).",
                            key="omit_feat", on_change=disable_widgets)

with st.empty():
    if len(st.session_state.omit_feat) > 0:
        st.info(f"The model can predict omitted features, bearing in mind that the accuracy may vary.", icon='📖')

with st.form("my_form"):
    cols = st.columns(4)
    with cols[0]:
        X[labels[0]] = st.slider(labels[0], min_value=0, max_value=120, value=50, disabled=st.session_state.omit_feat_mat[0])
        X[labels[1]] = st.slider(labels[1], min_value=0, max_value=200, value=76, disabled=st.session_state.omit_feat_mat[1])
        X[labels[2]] = st.select_slider(labels[2], options=[1.005, 1.010, 1.015, 1.020, 1.025], value=1.015, disabled=st.session_state.omit_feat_mat[2])
        X[labels[3]] = st.select_slider(labels[3], options=[0, 1, 2, 3, 4, 5], value=1, disabled=st.session_state.omit_feat_mat[3])
        X[labels[4]] = st.select_slider(labels[4], options=[0, 1, 2, 3, 4, 5], value=0, disabled=st.session_state.omit_feat_mat[4])

    with cols[1]:
        X[labels[5]] = st.selectbox(labels[5], ('Normal', 'Abnormal'), disabled=st.session_state.omit_feat_mat[5])
        X[labels[6]] = st.selectbox(labels[6], ('Normal', 'Abnormal'), disabled=st.session_state.omit_feat_mat[6])
        X[labels[7]] = st.selectbox(labels[7], ('Not Present', 'Present'), disabled=st.session_state.omit_feat_mat[7])
        X[labels[8]] = st.selectbox(labels[8], ('Not Present', 'Present'), disabled=st.session_state.omit_feat_mat[8])
        X[labels[9]] = st.slider(labels[9], min_value=0, max_value=500, value=150, disabled=st.session_state.omit_feat_mat[9])
        X[labels[10]] = st.slider(labels[10], min_value=0, max_value=400, value=60, disabled=st.session_state.omit_feat_mat[10])
        X[labels[11]] = st.slider(labels[11], min_value=0.0, max_value=80.0, value=3.1, step=0.1, disabled=st.session_state.omit_feat_mat[11])

    with cols[2]:
        X[labels[12]] = st.slider(labels[12], min_value=0.0, max_value=180.0, value=137.5, step=0.5, disabled=st.session_state.omit_feat_mat[12])
        X[labels[13]] = st.slider(labels[13], min_value=0.0, max_value=50.0, value=4.6, step=0.1, disabled=st.session_state.omit_feat_mat[13])
        X[labels[14]] = st.slider(labels[14], min_value=0.0, max_value=20.0, value=12.6, step=0.1, disabled=st.session_state.omit_feat_mat[14])
        X[labels[15]] = st.slider(labels[15], min_value=0, max_value=60, value=39, disabled=st.session_state.omit_feat_mat[15])
        X[labels[16]] = st.slider(labels[16], min_value=2000, max_value=26400, value=2600, step=10, disabled=st.session_state.omit_feat_mat[16])
        X[labels[17]] = st.slider(labels[17], min_value=2.0, max_value=10.0, value=4.7, step=0.1, disabled=st.session_state.omit_feat_mat[17])

    with cols[3]:
        X[labels[18]] = st.selectbox(labels[18], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[18])
        X[labels[19]] = st.selectbox(labels[19], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[19])
        X[labels[20]] = st.selectbox(labels[20], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[20])
        X[labels[21]] = st.selectbox(labels[21], ('Good', 'Poor'), disabled=st.session_state.omit_feat_mat[21])
        X[labels[22]] = st.selectbox(labels[22], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[22])
        X[labels[23]] = st.selectbox(labels[23], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[23])
    
    predict_btn = st.form_submit_button("Predict")

if predict_btn:
    # Handle omitted features
    X[st.session_state.omit_feat] = np.nan
    X_proc = X.copy()

    # Define the correct column names
    cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu',  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
            'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    rename_dict = {labels[i]: cols[i] for i in range(len(labels))}
    X_proc.rename(columns=rename_dict, inplace=True)

    # Apply preprocessing
    X_proc = X_proc.applymap(lambda s: s.lower().replace(' ', '') if isinstance(s, str) else s)

    # Load preprocessing objects
    with open(os.path.join(base_dir, 'assets', 'cat_imputer.pickle'), 'rb') as file:
        cat_imputer = pickle.load(file)

    with open(os.path.join(base_dir, 'assets', 'encoder.pickle'), 'rb') as file:
        encoder = pickle.load(file)

    with open(os.path.join(base_dir, 'assets', 'cont_imputer.pickle'), 'rb') as file:
        cont_imputer = pickle.load(file)

    with open(os.path.join(base_dir, 'assets', 'scaler.pickle'), 'rb') as file:
        scaler = pickle.load(file)

    with open(os.path.join(base_dir, 'assets', 'feat_extraction.pickle'), 'rb') as file:
        feat_extraction = pickle.load(file)

    with open(os.path.join(base_dir, 'assets', 'model.pickle'), 'rb') as file:
        model = pickle.load(file)

    # Preprocess the data
    X_proc[column_info['cat_imputer']] = cat_imputer.transform(X_proc[column_info['cat_imputer']])
    X_proc[column_info['encoder']] = encoder.transform(X_proc[column_info['encoder']])
    X_proc = cont_imputer.transform(X_proc)
    X_proc = pd.DataFrame(X_proc, columns=column_info['abbrev'])
    X_proc[column_info['scaler']] = scaler.transform(X_proc[column_info['scaler']])
    X_proc = feat_extraction.transform(X_proc)

    # Make prediction
    y_pred = model.predict(X_proc)

    # Display the result
    st.header("🎯 Prediction")
    if y_pred[0] == 1:
        st.error("The Patient has Chronic Kidney Disease (CKD).", icon='🩺')
    else:
        st.success("The Patient does not have Chronic Kidney Disease (CKD).", icon='🩺')
