import base64
import streamlit as st
from config import *
from streamlit_pages._home_page import home_page
from streamlit_pages._predict_alzheimer import prediction_page

# SETTING PAGE CONFIG
st.set_page_config(
    page_title="Alzheimer's Prediction System",
    page_icon=":brain:",
)

# Ensure CSS is correctly loaded
try:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
except NameError:
    st.error("CSS content not found. Ensure the file path is correct in the config.py.")

def set_page_background(png_file, default_file):
    @st.cache_data()
    def get_base64_of_bin_file(bin_file):
        try:
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except FileNotFoundError:
            return None
    
    bin_str = get_base64_of_bin_file(png_file)
    if not bin_str:
        bin_str = get_base64_of_bin_file(default_file)
    if bin_str:
        page_bg_img = f'''
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
            }}
            </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        st.error(f"Background image file not found: {png_file} and default file not found.")

set_page_background(BACKGROUND, DEFAULT_BACKGROUND)

# STREAMLIT APP
try:
    st.sidebar.image(SIDE_BANNER)
except FileNotFoundError:
    st.sidebar.error(f"Side banner image file not found: {SIDE_BANNER}")

st.sidebar.title("Alzheimer's Prediction System")
app_mode = st.sidebar.selectbox(
    "Please navigate through the different sections of our website from here",
    ["Home", "Predict Alzheimer's"],
)

st.sidebar.write("""
# Disclaimer
The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.

# Contact
For inquiries, you can mail us [here](mailto:aishwarya21824@gmail.com).
""")

def main():
    if app_mode == "Home":
        home_page()
    elif app_mode == "Predict Alzheimer's":
        prediction_page()

if __name__ == "__main__":
    main()
