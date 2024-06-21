import streamlit as st
st.set_page_config(
    page_title="LEAP",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

import pickle
import pandas as pd
    
# Navigation
st.page_link("pages/predict_page.py", label="Skip to Prediction", icon="🏹")

#loading Data
@st.cache_resource()
def load_model():
    with open('saved_model.pkl', 'rb') as file: 
        imported_data = pickle.load(file)
    
#caching data in to the session
if 'imported_data' not in st.session_state:
    imported_data = load_model()
    st.session_state.imported_data = imported_data

#Page display
def show_landing_page():
    
    st.title("LEAP Life Expectancy explore")

    st.write("""### select the country""")
    
    #inputs
    countriesList = st.session_state.imported_data["countriesList"]
    selectedCountry = st.selectbox("Country", countriesList)
    
    #getting recent data DF from the session
    Recent_data_DF =pd.DataFrame(st.session_state.imported_data["Recent_data_DF"])
    st.write(Recent_data_DF[Recent_data_DF['country'] == selectedCountry])
    
show_landing_page()



