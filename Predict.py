import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import time
import math

@st.cache(allow_output_mutation=True)
def pipeline(clf):
    # @st.cache ?
    census_df = pd.read_csv('https://drive.google.com/uc?id=1WRRikGZ_DCgioxm1NGORp7xa4itHRVcc&confirm=t&uuid=50742ec4-25a5-4196-b54e-38ea28882b71&at=AHV7M3cQOatyBQ8YtVlVBcNOSBX2:1670427300799')
    
    # Split 1: 80% Training, 20% Test
    split_census = ShuffleSplit(n_splits=1, random_state=237, test_size=0.20, train_size=None)
    for train_index, test_index in split_census.split(census_df, census_df['spm_resources']):
        census_train_set = census_df.iloc[train_index]
        census_test_set = census_df.iloc[test_index]   

    # Now we split the training set(80%) into a training set(60%) and validation set (20%)
    # Split 2: 60% Test; 20% Validation 
    split_census_2 = ShuffleSplit(n_splits=1, random_state=237, test_size=0.25, train_size=None)
    for train_index_6, validation_index in split_census_2.split(census_train_set, census_train_set['spm_resources']):
        census_train_set_6 = census_train_set.iloc[train_index_6]
        census_validation_set = census_train_set.iloc[validation_index] 

    census_X_train = census_train_set_6.drop('spm_resources', axis=1)
    census_y_train = census_train_set_6['spm_resources']

    census_X_valid = census_validation_set.drop('spm_resources', axis=1)
    census_y_valid = census_validation_set['spm_resources']

    census_X_test = census_test_set.drop('spm_resources', axis=1)
    census_y_test = census_test_set['spm_resources']
    
    clf.fit(census_X_train, census_y_train)
    return clf



#----------------------------------------------------------------

st.set_page_config(page_title="Predict")

user_inputs = st.container()
result = st.container()

with user_inputs:
    st.title("Predict")
    age_col, sex_col = st.columns(2)
    marital_stat, ed_level = st.columns(2)
    state, race = st.columns(2)

    user_age = age_col.number_input("Age:",min_value=21, max_value=67, value=20, step=1)
    user_sex = sex_col.selectbox("Sex:", ("Male", "Female"))
    user_marital = marital_stat.selectbox("Marital Status:", 
        ("Married", 
         "Widowed",
         "Divorced", 
         "Separated", 
         "Never married"))

    user_education = ed_level.selectbox("Education Level:", 
        ("Less than a high school degree",
        "High school degree",
        "Some college education",
        "College degree"))
    
    user_state = state.selectbox("State:",
        ("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", 
         "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", 
         "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
         "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
         "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"))
    
    user_race = race.selectbox("Race:", 
        ("White alone",
         "Black alone",
         "Asian alone",
         "Other (American Indian or Alaska Native, Pacific Islander, multiracial)"))
    
    #Converting to usuable data
    usuable_age = float(user_age)
    temp_sex = {"Male": 1.0, "Female": 2.0}
    usuable_sex = temp_sex[user_sex]

    temp_marital = {"Married": 1.0, 
                    "Widowed": 2.0,
                    "Divorced": 3.0, 
                    "Separated": 4.0, 
                    "Never married": 5.0}
    usuable_marital = temp_marital[user_marital]

    temp_education = {"Less than a high school degree": 1.0,
                        "High school degree": 2.0,
                        "Some college education": 3.0,
                        "College degree": 4.0}
    usuable_education = temp_education[user_education]

    #3, 7, 14, 43, 52
    temp_states = {"Alabama": 1.0, "Alaska": 2.0, "Arizona": 4.0, "Arkansas": 5.0, "California": 6.0, "Colorado": 8.0, "Connecticut": 9.0, "Delaware": 10.0, "District of Columbia": 11.0, "Florida": 12.0, 
         "Georgia": 13.0, "Hawaii": 15.0, "Idaho": 16.0, "Illinois": 17.0, "Indiana": 18.0, "Iowa": 19.0, "Kansas": 20.0, "Kentucky": 21.0, "Louisiana": 22.0, "Maine": 23.0, 
         "Maryland": 24.0, "Massachusetts": 25.0, "Michigan": 26.0, "Minnesota": 27.0, "Mississippi": 28.0, "Missouri": 29.0, "Montana": 30.0, "Nebraska": 31.0, "Nevada": 32.0, "New Hampshire": 33.0, 
         "New Jersey": 34.0, "New Mexico": 35.0, "New York": 36.0, "North Carolina": 37.0, "North Dakota": 38.0, "Ohio": 39.0, "Oklahoma": 40.0, "Oregon": 41.0, "Pennsylvania": 42.0, "Rhode Island": 44.0,
         "South Carolina": 45.0, "South Dakota": 46.0, "Tennessee": 47.0, "Texas": 48.0, "Utah": 49.0, "Vermont": 50.0, "Virginia": 51.0, "Washington": 53.0, "West Virginia": 54.0, "Wisconsin": 55.0, "Wyoming": 56.0}
    usuable_state = temp_states[user_state]

    temp_race = {"White alone": 1.0,
         "Black alone": 2.0,
         "Asian alone": 3.0,
         "Other (American Indian or Alaska Native, Pacific Islander, multiracial)": 4.0}
    usuable_race = temp_race[user_race]


with result:
    rf_model = RandomForestRegressor(random_state=23717125, verbose=2, n_jobs=8, max_depth=23, n_estimators=100)    
    #lin_reg = LinearRegression()
    v_df = pd.DataFrame(columns=['age', 'sex', 'mar', 'education', 'st', 'race'])
    v_df.loc[1] = [usuable_age, usuable_sex, usuable_marital, usuable_education, usuable_state, usuable_race]
    income_prediction = pipeline(rf_model).predict(v_df)
    st.title("Predicted income: $%.2f" % income_prediction[0])