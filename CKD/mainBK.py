import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# get data (from Kaggle):
df = pd.read_csv('kidney_disease.csv')

# DATA PREPROCESSING:
# Label encoding: Map text to 1/0 
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df.rename(columns={'classification':'class'},inplace=True)

# Data cleaning:
df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
df.drop('id',axis=1,inplace=True)
df2 = df.dropna(axis=0)
#df2['class'].value_counts()

# Data splitting:
X_train, X_test, y_train, y_test = train_test_split( df2.iloc[:,:-1], df2['class'],test_size = 0.33, random_state=44,stratify= df2['class'] )                                                     
                                                  
# Building Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,r2_score
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
# Model training:
dt.fit(X_train, y_train)

#dt_pred_train = dt.predict(X_train)
dt_pred_test = dt.predict(X_test)
score = r2_score(y_test,dt_pred_test)
fscore = f1_score(y_test,dt_pred_test)

################## Streamlit ###############

header = st.beta_container()

st.markdown("""
	<style>
	.main{
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
	""",
	unsafe_allow_html=True
)

with header:
    st.title('Predicting Chronic Kidney Disease')
    st.text('@Author: Thomas Nguyen Date: 12 March 2021')
    
    image = Image.open('ckd.png')
    st.image(image)
    
    st.text('CKD dataset from Kaggle: 400 x 25 features')
    df

    # plot some graphs:
    st.subheader('Age distribution of 250 people with hypertension plus CKD:')
    df1 = df[ df['htn']==1]
    df2 = df1[df1['class']==1 ]
    price_list = pd.DataFrame(df2['age'].round().value_counts())
    st.bar_chart(price_list)
    
    html_temp = """
	<div style="background-color:brown; padding:10px">
	<h2 style="color:white; text-align:center;">Machine Learning Model: Decision Tree</h2>
	</div>
	"""
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write('')

button_r2 = st.button('R2 score of the model')
st.write(score)
#if button_r2:
    #st.write(score)
button_f1 = st.button('F1 score of the model')
st.write(fscore)
#if button_f1:
    #st.write(fscore)

# Sidebar - Sector selection:
st.sidebar.header('Check yourself:')

age=st.sidebar.number_input('age',min_value=2, max_value=100, value=22, step=1, format=None, key=None)
bp=st.sidebar.number_input('blood pressure (mm/Hg)', min_value=45, max_value=180, value=66, step=1)
sg=st.sidebar.number_input('urin specific gravity (sg)',min_value=1.005,max_value=1.025,step=0.005)
al=st.sidebar.selectbox('albumin (al): yes (1) no (0)',options=[0,1,2,3,4,5])
su=st.sidebar.selectbox('sugar (su): yes (1) no (0)',options=[0,1,2,3,4,5])
rbc=st.sidebar.selectbox('red blood care (rbc): abnormal (1) normal (0)',options=[0,1])
pc=st.sidebar.selectbox('pus cell (pc): abnormal (1) normal (0)',options=[0,1])
pcc=st.sidebar.selectbox('pus cell clumps (pcc): present (1) not present (0)',options=[0,1])
ba=st.sidebar.selectbox('bacteria (ba): present (1) not present (0)',options=[0,1])
bgr=st.sidebar.number_input('blood gluco random (mgs/dl)',min_value=70, max_value=500, value=131, step=1)
bu=st.sidebar.number_input('blood urea (mgs/dl)',min_value=10, max_value=309, value=52, step=1)
sc=st.sidebar.number_input('serum creatinine (mgs/dl)',min_value=0.4, max_value=15.2, value=2.2, step=0.1)
sod=st.sidebar.number_input('sodium (mEq/L)',min_value=111, max_value=150, value=138, step=1)
pot=st.sidebar.number_input('potassium (mEq/L)',min_value=2.5, max_value=47.0, value=4.6, step=0.1)
hemo=st.sidebar.number_input('hemoglobin (gms)',min_value=3.1, max_value=17.8, value=13.7, step=0.1)
pcv=st.sidebar.number_input('packed cell count (pcv)',min_value=16, max_value=55, value=30, step=1)
wc=st.sidebar.number_input('white blood cell count (cells/cumm)',min_value=3000, max_value=15000, value=7000, step=100)
rc=st.sidebar.number_input('red blood cell count (millions/cumm)',min_value=2.2, max_value=6.9, value=5.0, step=0.1)
htn=st.sidebar.selectbox('hypertension (htn): yes (1) no (0)',options=[0,1])
dm=st.sidebar.selectbox('diabetes mellitus (dm): yes (1) no (0)',options=[0,1])
cad=st.sidebar.selectbox('coronary artery disease (cad): yes (1) no (0)',options=[0,1])
appet=st.sidebar.selectbox('appetite (appet): good (1) poor (0)',options=[0,1])
pe=st.sidebar.selectbox('pedal edema (pe): yes (1) no (0)',options=[0,1])
ane=st.sidebar.selectbox('anemia (ane): yes (1) no (0)',options=[0,1])

# make a query:
check_info = pd.DataFrame({'age':[age],'bp':[bp],'sg':[sg],'al':[al],'su':[su],'rbc':[rbc],'pc':[pc],'pcc':[pcc],'ba':[ba],'bgr':[bgr],'bu':[bu],'sc':[sc],'sod':[sod],'pot':[pot],'heml':[hemo],'pvc':[pcv],'wc':[wc],'rc':[rc],'htn':[htn],'dm':[dm],'cad':[cad],'appet':[appet],'pe':[pe],'ane':[ane]})
st.write('This is your given information:')
check_info
result = dt.predict(check_info)

#dd = df.loc[[df.index[335]]]  'pick one data row to test the model'
#dd = dd.iloc[:,:-1]
#dd
#result = dt.predict(dd)

if result==0:
    ans='You are not at risk of CKD.'
else:
    ans='You may be at risk of CKD. Check with your doctor.'

check_button = st.sidebar.button('Are you at risk?')
st.subheader('Model predicts:')
if check_button:
    st.sidebar.write(ans)
    st.write(ans)
