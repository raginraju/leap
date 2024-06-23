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

    
@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

countriesList = load_data('data/countriesList.csv')
Recent_data_DF  = load_data('data/Recent_data_DF.csv')
#st.dataframe(countriesList)
#st.dataframe(Recent_data_DF)
    

#Page display
def show_landing_page():
    
    st.title("LEAP Life Expectancy explore")

    st.write("""### select the country""")
    
    #inputs
    selectedCountry = st.selectbox("Country", countriesList)
    
    #getting recent data DF from the session
    st.write(Recent_data_DF[Recent_data_DF['country'] == selectedCountry])
    
show_landing_page()



