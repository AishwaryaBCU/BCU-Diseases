import os
import streamlit as st
from config import SIDE_BANNER, BG_IMAGE, DEFAULT_IMAGE

def main():
    st.title("Alzheimer's Prediction App")

    # Set up sidebar with image
    if os.path.isfile(SIDE_BANNER):
        st.sidebar.image(SIDE_BANNER)
    else:
        st.sidebar.error(f"Side banner image not found: {SIDE_BANNER}")

    # Set the background image if available
    if os.path.isfile(BG_IMAGE):
        set_page_background(BG_IMAGE)
    else:
        st.error(f"Background image not found: {BG_IMAGE}")

    # Call the home page function
    home_page()

def set_page_background(image_path):
    """Set the background image of the Streamlit page."""
    if os.path.isfile(image_path):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url({image_path});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Background image not found: {image_path}")

def home_page():
    """Render the home page content."""
    st.write("Welcome to the Alzheimer's Prediction App!")

if __name__ == "__main__":
    main()
