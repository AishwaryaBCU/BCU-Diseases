import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, r2_score
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('kidney_disease.csv')

# Data Preprocessing
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace({'yes': 1, 'no': 0})
df[['rbc','pc']] = df[['rbc','pc']].replace({'abnormal': 1, 'normal': 0})
df[['pcc','ba']] = df[['pcc','ba']].replace({'present': 1, 'notpresent': 0})
df[['appet']] = df[['appet']].replace({'good': 1, 'poor': 0, 'no': np.nan})
df['classification'] = df['classification'].replace({'ckd': 1.0, 'ckd\t': 1.0, 'notckd': 0.0, 'no': 0.0})
df.rename(columns={'classification': 'class'}, inplace=True)

# Data cleaning
df['pe'] = df['pe'].replace('good', 0)
df['appet'] = df['appet'].replace('no', 0)
df['cad'] = df['cad'].replace('\tno', 0)
df['dm'] = df['dm'].replace({'\tno': 0, '\tyes': 1, ' yes': 1, '': np.nan})
df.drop('id', axis=1, inplace=True)
df2 = df.dropna()

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:, :-1], df2['class'], test_size=0.33, random_state=44, stratify=df2['class'])

# Building and training Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

# Predictions and Scores
dt_pred_test = dt.predict(X_test)
score = r2_score(y_test, dt_pred_test)
fscore = f1_score(y_test, dt_pred_test)

################## Streamlit App ##################

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #0000ff;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: yellow;
        color: brown;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
with st.container():
    st.title('Predicting Chronic Kidney Disease')
    st.text('@Author: Thomas Nguyen Date: 12 March 2021')
    
    # Display image
    image = Image.open('ckd.png')
    st.image(image)
    
    st.text('CKD dataset from Kaggle: 400 x 25 features')
    st.write(df.head())  # Display first few rows of the dataset

    # Plot graphs
    st.subheader('Age distribution of people with hypertension and CKD:')
    df1 = df[df['htn'] == 1]
    df2 = df1[df1['class'] == 1]
    age_distribution = pd.DataFrame(df2['age'].round().value_counts())
    st.bar_chart(age_distribution)
    
    # Model description
    html_temp = """
    <div style="background-color:brown; padding:10px">
    <h2 style="color:white; text-align:center;">Machine Learning Model: Decision Tree</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

# Buttons to display scores
if st.button('Show R2 Score'):
    st.write(f"R2 score of the model: {score}")

if st.button('Show F1 Score'):
    st.write(f"F1 score of the model: {fscore}")

# Sidebar for user input
st.sidebar.header('Check yourself:')

# Collect user inputs
user_inputs = {
    'age': st.sidebar.number_input('Age', min_value=2, max_value=100, value=22),
    'bp': st.sidebar.number_input('Blood Pressure (mm/Hg)', min_value=45, max_value=180, value=66),
    'sg': st.sidebar.number_input('Urine Specific Gravity (sg)', min_value=1.005, max_value=1.025, step=0.005),
    'al': st.sidebar.selectbox('Albumin (al): Yes (1) No (0)', [0, 1, 2, 3, 4, 5]),
    'su': st.sidebar.selectbox('Sugar (su): Yes (1) No (0)', [0, 1, 2, 3, 4, 5]),
    'rbc': st.sidebar.selectbox('Red Blood Cell (rbc): Abnormal (1) Normal (0)', [0, 1]),
    'pc': st.sidebar.selectbox('Pus Cell (pc): Abnormal (1) Normal (0)', [0, 1]),
    'pcc': st.sidebar.selectbox('Pus Cell Clumps (pcc): Present (1) Not Present (0)', [0, 1]),
    'ba': st.sidebar.selectbox('Bacteria (ba): Present (1) Not Present (0)', [0, 1]),
    'bgr': st.sidebar.number_input('Blood Glucose Random (mg/dL)', min_value=70, max_value=500, value=131),
    'bu': st.sidebar.number_input('Blood Urea (mg/dL)', min_value=10, max_value=309, value=52),
    'sc': st.sidebar.number_input('Serum Creatinine (mg/dL)', min_value=0.4, max_value=15.2, value=2.2, step=0.1),
    'sod': st.sidebar.number_input('Sodium (mEq/L)', min_value=111, max_value=150, value=138),
    'pot': st.sidebar.number_input('Potassium (mEq/L)', min_value=2.5, max_value=47.0, value=4.6, step=0.1),
    'hemo': st.sidebar.number_input('Hemoglobin (g/dL)', min_value=3.1, max_value=17.8, value=13.7, step=0.1),
    'pcv': st.sidebar.number_input('Packed Cell Volume (pcv)', min_value=16, max_value=55, value=30),
    'wc': st.sidebar.number_input('White Blood Cell Count (cells/cumm)', min_value=3000, max_value=15000, value=7000, step=100),
    'rc': st.sidebar.number_input('Red Blood Cell Count (millions/cumm)', min_value=2.2, max_value=6.9, value=5.0, step=0.1),
    'htn': st.sidebar.selectbox('Hypertension (htn): Yes (1) No (0)', [0, 1]),
    'dm': st.sidebar.selectbox('Diabetes Mellitus (dm): Yes (1) No (0)', [0, 1]),
    'cad': st.sidebar.selectbox('Coronary Artery Disease (cad): Yes (1) No (0)', [0, 1]),
    'appet': st.sidebar.selectbox('Appetite (appet): Good (1) Poor (0)', [0, 1]),
    'pe': st.sidebar.selectbox('Pedal Edema (pe): Yes (1) No (0)', [0, 1]),
    'ane': st.sidebar.selectbox('Anemia (ane): Yes (1) No (0)', [0, 1])
}

# Create DataFrame for prediction
check_info = pd.DataFrame([user_inputs])

# Make a prediction
result = dt.predict(check_info)

# Display result
if st.sidebar.button('Are you at risk?'):
    if result[0] == 0:
        ans = 'You are not at risk of CKD.'
    else:
        ans = 'You may be at risk of CKD. Check with your doctor.'
    st.write(ans)
