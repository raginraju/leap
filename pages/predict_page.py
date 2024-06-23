import streamlit as st
import pickle
import numpy as np
import pandas as pd

def intializeArray():
    return np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
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
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])


@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

countriesList = load_data('data/countriesList.csv')
predictors  = load_data('data/predictors.csv')
Recent_data_DF  = load_data('data/Recent_data_DF.csv')
#st.dataframe(countriesList)
#st.dataframe(predictors)

@st.cache_resource
def load_model():
    with open('saved_model.pkl', 'rb') as file: 
        saved_model = pickle.load(file)
    return saved_model

model = load_model()
regressor = model["model"]

# Navigation
st.page_link("app.py", label="Go to Home", icon="🏠")


def show_predict_page():
    X = intializeArray()

    st.title("LEAP Life Expectancy Prediction")

    st.write("""### We need some information to predict the result""")

    #inputs
    selectedCountry = st.selectbox("Country", countriesList)
    selected_country_data = Recent_data_DF[Recent_data_DF['country'] == selectedCountry]

    recentYear = int(selected_country_data["year"])
    year = st.number_input("year", value = recentYear)
    
    recentPopulation = int(selected_country_data["population"])
    population = st.slider("Population", min_value=recentPopulation, max_value=1424929792,value=recentPopulation)

    recentAnnualCo2 = float(selected_country_data["annual_co2_emissions_scaled"])
    AnnualCo2 = st.number_input("AnnualCo2_scaled", value=recentAnnualCo2)
    
    recentAnimalProtein = float(selected_country_data["animal_protein_per_day_person_scaled"])
    AnimalProtein = st.number_input("Animal Protien per day scaled", value=recentAnimalProtein)
    
    recentFat = float(selected_country_data["fat_per_day_person_scaled"])
    Fat = st.number_input("Fat Per Day", value=recentFat)
    
    recentCarb = float(selected_country_data["carbs_per_day_person_scaled"])
    Carb = st.number_input("Carb Per Day", value=recentCarb )
    
    recentFertility = float(selected_country_data["fertility_rate_scaled"])
    Fertility = st.number_input("Fertility Rate", value=recentFertility)
    
    recentRuralPop = float(selected_country_data["rural_population_by_pop_scaled"])
    RuralPop = st.number_input("Rural Population" , value=recentRuralPop)
    
    recentFixedLineSub = float(selected_country_data["fixed_line_subscription_per_hundred_scaled"])
    FixedLineSub = st.number_input("Fixed Line Subscription", value=recentFixedLineSub)
    
    recentMobileLineSub = float(selected_country_data["mobile_line_subscription_per_hundred_scaled"])
    MobileLineSub = st.number_input("Moblie line Subscription", value=recentMobileLineSub)

    ok = st.button("Predict life Expectancy")
    
    if ok:
        
        #assigning to input array
        #If afghanistan; set all countries to zero
        if (selectedCountry != "Afghanistan"): 
            countryColName = "country_"+selectedCountry
            countryIndex = predictors[predictors.iloc[:, 0]==countryColName].index
            #st.text(predictors[predictors.iloc[:, 0]==countryColName].index)
            X[0,countryIndex] = 1
        X[0,0] = year
        X[0,1] = population
        X[0,2] = AnnualCo2
        X[0,3] = AnimalProtein 
        X[0,4] = Fat
        X[0,5] = Carb
        X[0,6] = Fertility
        X[0,7] = RuralPop
        X[0,8] = FixedLineSub
        X[0,9] = MobileLineSub
        
        X = X.astype(float)
        
        life_exp = regressor.predict(X)
        
        #st.text(f"Selected Country {selectedCountry} ")
        #st.text(f"Selected Year {year} ")
        #st.text(f"Selected Population {population} ")
        st.subheader(f"The calculated Life Expectancy is {life_exp[0]:.2f} ")
        
        #Reset county value to zero after each prediction
        X = intializeArray()



show_predict_page()
