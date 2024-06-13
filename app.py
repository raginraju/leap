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
#from pages.predict_page import show_predict_page

def load_model():
    with open('saved_model.pkl', 'rb') as file: 
        data = pickle.load(file)
    return data

if 'data' not in st.session_state:
    data = load_model()
    st.session_state['data'] = data


def show_landing_page():

    #data = load_model()
    
    countriesList = st.session_state.data["countriesList"]
    Recent_data = st.session_state.data["Recent_data_DF"]
    Recent_data_DF = pd.DataFrame(Recent_data)

    st.title("LEAP Life Expectancy explore")

    st.write("""### select the country""")

    #inputs
    selectedCountry = st.selectbox("Country", countriesList)
    st.write(Recent_data_DF[Recent_data_DF['country'] == selectedCountry])
    
# Navigation
st.page_link("pages/predict_page.py", label="Skip to Prediction", icon="🏹")

#Page display
show_landing_page()



