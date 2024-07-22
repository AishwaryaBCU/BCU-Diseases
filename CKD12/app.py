from streamlit_option_menu import option_menu  # Import option_menu

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Health Assistant',
                           ['CKD Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity'],
                           default_index=0)

# CKD Prediction Page
if selected == "CKD Prediction":
    st.title('Chronic Kidney Disease (CKD) Prediction using ML')
    # (Include the rest of your CKD prediction code here)
