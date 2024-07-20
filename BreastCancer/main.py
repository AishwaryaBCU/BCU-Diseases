import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import base64

def get_clean_data():
    data_file = "BreastCancer/data.csv"  # Updated path
    if not os.path.exists(data_file):
        st.error(f"File `{data_file}` not found. Please ensure it is in the correct directory.")
        return pd.DataFrame()
    
    try:
        data = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Error reading `{data_file}`: {e}")
        return pd.DataFrame()

    try:
        data = data.drop(['Unnamed: 32', 'id'], axis=1)
    except KeyError as e:
        st.error(f"Column error: {e}. Please check the column names in `{data_file}`.")
        return pd.DataFrame()

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_background_image():
    bg_image_file = "BreastCancer/bg.webp"  # Updated path
    if not os.path.exists(bg_image_file):
        st.error(f"Background image file `{bg_image_file}` not found.")
        return

    try:
        with open(bg_image_file, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/webp;base64,{encoded_image}");
                    background-size: cover;
                    background-position: center;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error loading background image `{bg_image_file}`: {e}")

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    if data.empty:
        return {}

    slider_labels = [
        # ... [list of sliders]
    ]

    input_dict = {}
    for label, key in slider_labels:
        if key in data.columns:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
        else:
            st.warning(f"Column `{key}` is missing from `{data_file}`.")
            input_dict[key] = 0

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    if data.empty:
        return {}

    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        if key in X.columns:
            max_val = X[key].max()
            min_val = X[key].min()
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_dict[key] = scaled_value
        else:
            st.warning(f"Column `{key}` is missing for scaling.")
            scaled_dict[key] = 0

    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
          r=[
            input_data.get('radius_mean', 0), input_data.get('texture_mean', 0), input_data.get('perimeter_mean', 0),
            input_data.get('area_mean', 0), input_data.get('smoothness_mean', 0), input_data.get('compactness_mean', 0),
            input_data.get('concavity_mean', 0), input_data.get('concave points_mean', 0), input_data.get('symmetry_mean', 0),
            input_data.get('fractal_dimension_mean', 0)
          ],
          theta=categories,
          fill='toself',
          name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
          r=[
            input_data.get('radius_se', 0), input_data.get('texture_se', 0), input_data.get('perimeter_se', 0), input_data.get('area_se', 0),
            input_data.get('smoothness_se', 0), input_data.get('compactness_se', 0), input_data.get('concavity_se', 0),
            input_data.get('concave points_se', 0), input_data.get('symmetry_se', 0),input_data.get('fractal_dimension_se', 0)
          ],
          theta=categories,
          fill='toself',
          name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
          r=[
            input_data.get('radius_worst', 0), input_data.get('texture_worst', 0), input_data.get('perimeter_worst', 0),
            input_data.get('area_worst', 0), input_data.get('smoothness_worst', 0), input_data.get('compactness_worst', 0),
            input_data.get('concavity_worst', 0), input_data.get('concave points_worst', 0), input_data.get('symmetry_worst', 0),
            input_data.get('fractal_dimension_worst', 0)
          ],
          theta=categories,
          fill='toself',
          name='Worst Value'
    ))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True
    )
    return fig

def add_predictions(input_data):
    model_file = "BreastCancer/model.pkl"  # Updated path
    scaler_file = "BreastCancer/scaler.pkl"  # Updated path
    
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        st.error(f"Model file `{model_file}` or scaler file `{scaler_file}` not found.")
        return

    try:
        model = pickle.load(open(model_file, "rb"))
        scaler = pickle.load(open(scaler_file, "rb"))
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.write("The predicted cell cluster is:")
    if prediction[0] == 0:
        st.write("Benign", unsafe_allow_html=True)
    else:
        st.write("Malicious", unsafe_allow_html=True)

    st.write("Probability of Benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of Malicious: ", model.predict_proba(input_array_scaled)[0][1])

def list_files_in_directory():
    cwd = os.getcwd()
    st.write(f"Current working directory: `{cwd}`")
    files = os.listdir(cwd)
    st.write("Files in the current directory:")
    st.write(files)

def main():
    st.set_page_config(
        page_title="Breast Cancer Diagnosis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Display the current working directory and list files
    list_files_in_directory()

    add_background_image()
    input_data = add_sidebar()
    if input_data:
        with st.container():
            st.title("Breast Cancer Diagnosis")
            st.write("This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
        col1, col2 = st.columns([4, 1])
        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
        with col2:
            add_predictions(input_data)

if __name__ == '__main__':
    main()
