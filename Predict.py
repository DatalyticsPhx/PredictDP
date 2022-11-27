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
import time
import math

st.set_page_config(page_title="Predict")

user_inputs = st.container()
result = st.container()

census_2016 = pd.read_sas("C:/Users/VTech/CTP/spm_pu_2016.sas7bdat")

split_census = ShuffleSplit(n_splits=1, random_state=237, test_size=0.20, train_size=None)
for train_index, test_index in split_census.split(census_2016, census_2016['agi']):
    census_train_set = census_2016.iloc[train_index]
    census_test_set = census_2016.iloc[test_index]   

census_X_train = census_train_set.drop('agi', axis=1)
census_y_train = census_train_set['agi']

census_X_test = census_test_set.drop('agi', axis=1)
census_y_test = census_test_set['agi'] 
   
lin_reg = LinearRegression()
lin_reg.fit(census_X_train, census_y_train)

v_df = pd.DataFrame(columns=test_census.columns)
v_df.loc[1] = [66.0, 2.0, 1.0, 3.0, 47.0, 1.0]
pred = lin_reg.predict(v_df)

with user_inputs:
    st.title("Predict")
    age_col, sex_col = st.columns(2)
    marital_stat, ed_level = st.columns(2)
    state, race = st.columns(2)

    user_age = age_col.number_input("Age:",min_value=0, max_value=100, value=20, step=1)
    user_sex = sex_col.selectbox("Sex:", ("Male", "Female"))
    user_marital = marital_stat.selectbox("Marital Status:", 
        ("Married", 
         "Widowed",
         "Divorced", 
         "Separated", 
         "Never married"))

    user_education = ed_level.selectbox("Education Level:", 
        ("Under age 25/NIU",
        "Less than a high school degree",
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

with result:
    st.title("Predicted income:" + str(pred))