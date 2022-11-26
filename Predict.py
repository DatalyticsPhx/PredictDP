import streamlit as st
st.set_page_config(page_title="Predict")

user_inputs = st.container()
result = st.container()


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
    st.title("Predicted income:")