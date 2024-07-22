import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# get data:
df = pd.read_csv('kidney_disease.csv')

# Label encoding: Map text to 1/0 and do some cleaning
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1,'ckd\t':1,'notckd':0,'no':0})
df.rename(columns={'classification':'class'},inplace=True)

# Data preprocessing:
df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
df.drop('id',axis=1,inplace=True)

# Drop rows with missing data:
df2 = df.dropna(axis=0)
df2['class'].value_counts() # 43 for 1 and 115 for 0 (unbalanced dataset)

# Solve unbalanced data problem: use SMOTH to increase data for 1:
from imblearn.over_sampling import SMOTE
X = df2.iloc[:,:-1]
y = df2['class']
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
# create data for seaborn scatter plot:
data_sns = pd.concat([X_resampled,y_resampled],axis=1)
data_sns['class'] = data_sns['class'].replace(to_replace={1:'ckd',0:'notckd'})

# Data splitting:
# X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:,:-1], df2['class'], test_size = 0.33, stratify = df2['class'] )
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.33, stratify = y_resampled )

# Load Decision Tree model:
from sklearn.metrics import f1_score, r2_score, confusion_matrix, accuracy_score
loaded_model = pickle.load(open('CKDdecisiontree_model.sav','rb'))

dt_pred_test = loaded_model.predict(X_test)

score = r2_score(y_test,dt_pred_test)
fscore = f1_score(y_test,dt_pred_test)
cm = confusion_matrix(y_test,dt_pred_test)
accuracy = accuracy_score(y_test,dt_pred_test)

################## Streamlit ###############
st.title('_Big Brother messed up my data. My apps are not working properly_ :sunglasses:')

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
    st.text('@Author: Thomas Nguyen Date: 28 Feb 2021')
    image = Image.open('ckd.png')
    st.image(image)
	
    st.text('CKD dataset from Kaggle: 400 x 25 features')
    df
    st.markdown(''':red[Big Brother messed up my data...All of my apps are not working properly.]''')
	
     # plot some graphs:
    st.subheader('Hypertension and Diabetes vs. CKD:')
    dfa = df[ df['htn']==0 ]
    dfb = dfa[ dfa['dm']==0 ]
    t1 = pd.DataFrame.sum(dfb['age'].round().value_counts())
    dfc = dfb[dfb['class']==1 ]
    t2 = pd.DataFrame.sum(dfc['age'].round().value_counts())
    ratio1 = t2/t1

    dfA = df[ df['htn']==1 ]
    dfB = dfA[ dfA['dm']==1 ]
    tt1 = pd.DataFrame.sum(dfB['age'].round().value_counts())
    dfC = dfB[dfB['class']==1 ]
    tt2 = pd.DataFrame.sum(dfC['age'].round().value_counts())
    ratio2 = tt2/tt1
    
    st.write('Group 1: People got CKD without hypertension and diabetes symptoms')
    st.write('Group 2: People got CKD with hypertension and diabetes symptoms')

    fig, ax = plt.subplots()
    # set width of bars
    barWidth = 0.25
 
    # set heights of bars
    bars1 = [ratio1]
    bars2 = [ratio2]
 
    # Set position of bar on X axis
    #r1 = np.arange(len(bars1))
    #r2 = [x + barWidth for x in r1]
    r1 = 0.75
    r2 = 1.25
 
    # Make the plot
    plt.bar(r1, bars1, color='#00ff00', width=barWidth, edgecolor='white', label='group 1')
    plt.bar(r2, bars2, color='#ff0000', width=barWidth, edgecolor='white', label='group 2')
 
    # Add xticks on the middle of the group bars
    plt.xlim([0,2])
    plt.ylabel('Percentage (100%)', fontweight='bold')
    #plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
 
    # Create legend & Show graphic
    plt.legend()
    st.pyplot(fig,use_column_width=True)
    
    html_temp = """
	<div style="background-color:brown; padding:10px">
	<h2 style="color:white; text-align:center;">Machine Learning Model Performance</h2>
	</div>
	"""
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write('')

fig2 = plt.figure()
sns.scatterplot(data=data_sns , x = 'age' , y = 'hemo' , hue='class')
st.pyplot(fig2)

st.write('Confusion Matrix: (balanced dataset)', cm)
st.write('')
#button_r2 = st.button('R2 score of the model')
#st.write(score)

#button_f1 = st.button('F1 score of the model')

st.write('F1 score of the model:')
st.write(fscore)

#button_acc = st.button('Accuracy score of the model')
st.write('Accuracy score of the model:')
st.write(accuracy)

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
st.write('This is your given information: (you can modify this information from the left sidebar)')
check_info
result = loaded_model.predict(check_info)

if result==0:
    ans='You are not at risk of CKD.'
else:
    ans='You might be at risk of CKD. Check with your doctor.'

st.write('')
check_button = st.button('Check to see if you are at risk of KD?')
st.subheader('Model predicts:')
if check_button:
    st.sidebar.write(ans)
    st.write(ans)
