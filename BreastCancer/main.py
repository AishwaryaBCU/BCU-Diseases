import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Function to get cleaned data
def get_clean_data():
    data_path = "BreastCancer/data.csv"
    if not os.path.exists(data_path):
        st.error(f"File {data_path} not found.")
        return pd.DataFrame()  # Return an empty DataFrame if file not found

    data = pd.read_csv(data_path)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# Function to add sidebar with input sliders
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()
    if data.empty:
        st.warning("Data is not available.")
        return {}

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
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
            st.warning(f"Column `{key}` is missing from the data.")
            input_dict[key] = 0  # Default value or handle as appropriate

    st.read("Sidebar input values:", input_dict)  # Debugging line
    return input_dict

# Function to scale input values
def get_scaled_values(input_dict):
    data = get_clean_data()
    if data.empty:
        return {key: 0 for key in input_dict}  # Return default values if data is empty

    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        if key in X.columns:
            max_val = X[key].max()
            min_val = X[key].min()
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_dict[key] = scaled_value
        else:
            scaled_dict[key] = 0  # Default scaling

    return scaled_dict

# Function to get radar chart
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
            input_data.get('radius_se', 0), input_data.get('texture_se', 0), input_data.get('perimeter_se', 0),
            input_data.get('area_se', 0), input_data.get('smoothness_se', 0), input_data.get('compactness_se', 0),
            input_data.get('concavity_se', 0), input_data.get('concave points_se', 0), input_data.get('symmetry_se', 0),
            input_data.get('fractal_dimension_se', 0)
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

# Function to add predictions
def add_predictions(input_data):
    model_file = "BreastCancer/model.pkl"
    scaler_file = "BreastCancer/scaler.pkl"

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        st.error(f"Model file `{model_file}` or scaler file `{scaler_file}` not found.")
        return

    try:
        model = pickle.load(open(model_file, "rb"))
        scaler = pickle.load(open(scaler_file, "rb"))
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return

    st.read("Input Data for Prediction:", input_data)  # Debugging line

    try:
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
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Main function to run the app
def main():
    st.set_page_config(
        page_title="Breast Cancer Diagnosis",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()

    if not input_data:
        return  # Exit if no valid input data

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
