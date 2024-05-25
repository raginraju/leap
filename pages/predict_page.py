import streamlit as st
import pickle
import numpy as np

def intializeArray():
    return np.array([[1.98500000e+03, 6.24554200e+06, 6.31222871e-05, 1.04519190e-01,
        2.63156492e-01, 3.96935618e-01, 3.30483972e-01, 7.65390877e-01,
        5.34068911e-02, 3.67066458e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

def load_model():
    with open('../saved_steps.pkl', 'rb') as file: 
        data = pickle.load(file)
    return data



def show_predict_page():
    X = intializeArray()

    data = load_model()

    regressor = data["model"]
    countriesList = data["countriesList"]
    predictors = data["predictors"]

    st.title("LEAP Life Expectancy Prediction")

    st.write("""### We need some information to predict the result""")

    #inputs
    country = st.selectbox("Country", countriesList)
    year = st.slider("Year", min_value=2024, max_value=2070)
    population = st.slider("Population", min_value=5000000, max_value=1424929792)

    ok = st.button("Predict life Expectancy")
    
    if ok:
        #assigning to input array
        #If afghanistan; set all countries to zero
        if (country != "Afghanistan"): 
            countryColName = "country_"+country
            countryIndex = predictors.index(countryColName)
            X[0,countryIndex] = 1
        X[0,0] = year
        X[0,1] = population
        
        X = X.astype(float)
        
        life_exp = regressor.predict(X)
        
        st.subheader(f"The calculated Life Expectancy is {life_exp[0]:.2f}")
