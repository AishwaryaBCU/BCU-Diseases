import os
import streamlit as st

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets', 'images')

# Define image paths
SIDE_BANNER = os.path.join(ASSETS_DIR, 'side_banner.webp')
BG_IMAGE = os.path.join(ASSETS_DIR, 'bg.webp')
DEFAULT_IMAGE = os.path.join(ASSETS_DIR, 'default.webp')

# Load secrets
NEWS_API_KEY = st.secrets.get("NEWS_API", None)  # Ensure you handle missing secrets properly
