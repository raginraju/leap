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
from pages.predict_page import show_predict_page

def load_model():
    with open('saved_model.pkl', 'rb') as file: 
        data = pickle.load(file)
    return data


def show_landing_page():

    data = load_model()
    
    countriesList = data["countriesList"]
    Recent_data_DF = data["Recent_data_DF"]

    st.title("LEAP Life Expectancy explore")

    st.write("""### select the country""")

    #inputs
    country = st.selectbox("Country", countriesList)
    st.write("Life expectancy of the selected country in 2020 is ")
    
# Navigation
st.page_link("pages/predict_page.py", label="Skip to Prediction", icon="🏹")

show_landing_page()


