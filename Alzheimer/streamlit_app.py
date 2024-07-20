import base64
import streamlit as st
from config import *
from streamlit_pages._home_page import home_page
from streamlit_pages._predict_alzheimer import prediction_page
from streamlit_pages._latest_news import news_page
from streamlit_pages._team_members import team_members  
from streamlit_pages._chat_page import chat_bot

# SETTING PAGE CONFIG
st.set_page_config(
    page_title="Alzheimer's Prediction Systems",
    page_icon=":brain:",
)

# Apply custom CSS
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

def set_page_background(png_file):
    @st.cache_data()
    def get_base64_of_bin_file(bin_file):
        try:
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except FileNotFoundError:
            st.error(f"Background image file not found: {bin_file}")
            return None
    
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f'''
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-position: center;
            }}
            </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the page background
set_page_background(BACKGROUND) 

# STREAMLIT APP
if SIDE_BANNER:
    if os.path.isfile(SIDE_BANNER):
        st.sidebar.image(SIDE_BANNER)
    else:
        st.sidebar.error(f"Side banner image file not found: {SIDE_BANNER}")

st.sidebar.title("Alzheimer's Prediction System")
app_mode = st.sidebar.selectbox(
    "Please navigate through the different sections of our website from here",
    ["Home", "Predict Alzheimer's", "ChatBot", "Latest News", "Team Members"],
)

st.sidebar.write("""
# Disclaimer
The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.

# Contact
For inquiries, you can mail us [here](mailto:arpitsengar99@gmail.com).
""")

def main():
    if app_mode == "Home":
        home_page()
    elif app_mode == "Predict Alzheimer's":
        prediction_page()
   

if __name__ == "__main__":
    main()
