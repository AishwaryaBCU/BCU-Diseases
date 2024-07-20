import os
import streamlit as st
import base64

# SETTING PAGE CONFIG
st.set_page_config(
    page_title="Alzheimer's Prediction Systems",
    page_icon=":brain:",
)

# Print current working directory
st.write("Current working directory:", os.getcwd())

# Define paths using absolute paths
base_path = os.path.dirname(os.path.abspath(__file__))
bg_image_path = os.path.join(base_path, 'assets', 'images', 'bg.webp')
default_image_path = os.path.join(base_path, 'assets', 'images', 'default.webp')
side_banner_path = os.path.join(base_path, 'assets', 'images', 'side_banner.webp')

# List files in the assets/images directory
files = os.listdir(os.path.join(base_path, 'assets', 'images'))
st.write("Files in 'assets/images' directory:", files)

# Check if images exist
if os.path.isfile(bg_image_path):
    st.write(f"Background image found: {bg_image_path}")
else:
    st.error(f"Background image not found: {bg_image_path}")

if os.path.isfile(default_image_path):
    st.write(f"Default image found: {default_image_path}")
else:
    st.error(f"Default image not found: {default_image_path}")

if os.path.isfile(side_banner_path):
    st.write(f"Side banner image found: {side_banner_path}")
else:
    st.error(f"Side banner image not found: {side_banner_path}")

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
            }}
            </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

# Try setting the page background with the absolute path
set_page_background(bg_image_path)

# STREAMLIT APP
if os.path.isfile(side_banner_path):
    st.sidebar.image(side_banner_path)

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
        from streamlit_pages._home_page import home_page
        home_page()
    elif app_mode == "Predict Alzheimer's":
        from streamlit_pages._predict_alzheimer import prediction_page
        prediction_page()

if __name__ == "__main__":
    main()
