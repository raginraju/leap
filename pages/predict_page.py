import streamlit as st
import pickle
import numpy as np
import pandas as pd
import scaler

def intializeArray():
    return np.array([[0.00000000e+00, 0.00000000e+00, 
                      0.00000000e+00,0.00000000e+00,0.00000000e+00, 
                      ###Pareto Start #########
                      #0.00000000e+00,0.00000000e+00, 0.00000000e+00, 0.00000000e+00
                      ###Pareto End #########
                      ]])

@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

countriesList = load_data('data/countriesList.csv')
predictors  = load_data('data/predictors.csv')
Recent_data_DF  = load_data('data/Recent_data_DF.csv')


@st.cache_resource
def load_model():
    with open('saved_final_RFParetomodel.pkl', 'rb') as file: 
        saved_model = pickle.load(file)
    return saved_model

model = load_model()
regressor = model["model"]

# Navigation
st.page_link("app.py", label="Go to Home", icon="🏠")


def show_predict_page():
    X = intializeArray()

    st.title("LEAP Life Expectancy Prediction")

    #st.write("""### We need some information to predict the result""")

    #inputs
    selectedCountry = st.selectbox("Country", countriesList)
    selected_country_data = Recent_data_DF[Recent_data_DF['country'] == selectedCountry]
    
    # recentPopulation = (selected_country_data["population"].iloc[0])
    # population = st.number_input("Population", value=recentPopulation, step=1000.00)

    recentFixedLineSub = (selected_country_data["fixed_line_subscription_per_hundred_scaled"].iloc[0])
    FixedLineSub = st.number_input("Fixed Line Subscription Per Hundred", value=scaler.reverse_min_max_scaling(recentFixedLineSub,"FIXEDLINE"), step=1.00)
    
    recentFertility = (selected_country_data["fertility_rate_scaled"].iloc[0])
    Fertility = st.number_input("Fertility Rate", value=scaler.reverse_min_max_scaling(recentFertility,"FERTILITY"), step=1.00)
    
    recentFat = (selected_country_data["fat_per_day_person_scaled"].iloc[0])
    Fat = st.number_input("Fat Per Day Person", value=scaler.reverse_min_max_scaling(recentFat,"FAT"), step=100.00)
    
    recentCarb = (selected_country_data["carbs_per_day_person_scaled"].iloc[0])
    Carb = st.number_input("Carbs Per Day Person", value=scaler.reverse_min_max_scaling(recentCarb,"CARB"), step=100.00 )
    
    ###Pareto Start #########
    # recentMobileLineSub = (selected_country_data["mobile_line_subscription_per_hundred_scaled"].iloc[0])
    # MobileLineSub = st.number_input("Moblie Line Subscription Per Hundred", value=scaler.reverse_min_max_scaling(recentMobileLineSub,"MOBILELINE"))
    
    # recentAnnualCo2 = (selected_country_data["annual_co2_emissions_scaled"].iloc[0])
    # AnnualCo2 = st.number_input("Annual CO2 Emissions", value=scaler.reverse_min_max_scaling(recentAnnualCo2,"ANNUALCO2") , step=100000.00)
    
    # recentRuralPop = (selected_country_data["rural_population_by_pop_scaled"].iloc[0])
    # RuralPop = st.number_input("Rural Population by Population" , value=scaler.reverse_min_max_scaling(recentRuralPop,"RURAL"))
    
    # recentAnimalProtein = (selected_country_data["animal_protein_per_day_person_scaled"].iloc[0])
    # AnimalProtein = st.number_input("Animal Protien Per Day Person", value=scaler.reverse_min_max_scaling(recentAnimalProtein, "ANIMALPROTIEN"))
    
    ###Pareto End #########

    ok = st.button("Predict life Expectancy")
    
    if ok:
        
        #assigning to input array
        X[0,0] = population
        X[0,1] = scaler.min_max_scaling(Fat,"FAT")
        X[0,2] = scaler.min_max_scaling(Carb,"CARB") 
        X[0,3] = scaler.min_max_scaling(Fertility,"FERTILITY")
        X[0,4] = scaler.min_max_scaling(FixedLineSub,"FIXEDLINE")
        
        ###Pareto Start #########
        # X[0,0] = population
        # X[0,1] = scaler.min_max_scaling(AnnualCo2, "ANNUALCO2")
        # X[0,2] = scaler.min_max_scaling(AnimalProtein,"ANIMALPROTIEN") 
        # X[0,3] = scaler.min_max_scaling(Fat,"FAT")
        # X[0,4] = scaler.min_max_scaling(Carb,"CARB")
        # X[0,5] = scaler.min_max_scaling(Fertility,"FERTILITY")
        # X[0,6] = scaler.min_max_scaling(RuralPop,"RURAL")
        # X[0,7] = scaler.min_max_scaling(FixedLineSub,"FIXEDLINE")
        # X[0,8] = scaler.min_max_scaling(MobileLineSub,"MOBILELINE")
        ###Pareto End #########
        
        X = X.astype(float)
        
        life_exp = regressor.predict(X)
        
        #st.text(f"Selected Country {selectedCountry} ")
        #st.text(f"Selected Year {year} ")
        #st.text(f"Selected Population {population} ")
        #print(f"The calculated Life Expectancy is {life_exp[0]:.2f} ")
        st.subheader(f"The calculated Life Expectancy is {life_exp[0]:.2f} ")
        
        #Reset county value to zero after each prediction
        #X = intializeArray()



show_predict_page()
