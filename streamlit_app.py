# Import necessary libraries
import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.metrics import confusion_matrix
import pickle
import joblib
from sklearn.linear_model import LogisticRegression


@st.cache_resource
def loadModels():
        # Load the models dictionary
    filename = 'model/Diabetic-Risk-Prediction-models.pkl'
    return joblib.load(open(filename, 'rb'))

st.subheader("Diabetic Risk Prediction")
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("""
    To predict your Diabetic status, simply follow the steps below:
    1. Enter the parameters that best describe you;
    2. Press the "Predict" button and wait for the result.
    """)
with col2:
    st.image("images/heart-sidebar.png", width=100)

submit = st.button("Predict")


# st.write(BMIdata)

# Thesidebar func tion from streamlit is used to create a sidebar for users 
# to input their information.
# --------------------------------------------------------------------------
st.sidebar.title('Please, fill your informations to predict your heart condition')

Gender=st.sidebar.selectbox("Select your gender", ("Female", 
                             "Male" ))

Age=st.sidebar.selectbox("Select your age", 
                            ("18-24", 
                             "25-29",
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "55-59",
                             "80 or older"))

BMI=st.sidebar.number_input("BMI",18,100,18)

Smoking = st.sidebar.selectbox("Have you smoked more than 100 cigarettes in"
                          " your entire life ?)",
                          options=("No", "Yes"))

genHealth = st.sidebar.selectbox("General health",
                             options=("Good","Excellent", "Fair", "Very good", "Poor"))
physAct = st.sidebar.selectbox("Physical activity in the past month"
                           , options=("No", "Yes"))

asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))

kidneyDisease= st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))

income = st.sidebar.selectbox("What is your Income?",options=("$75,000 or more", "Less than $10,000",
                                          "Less than $15,000 ($10,000 to less than $15,000)"
                                          "Less than $20,000 ($15,000 to less than $20,000)",
                                          "Less than $25,000 ($20,000 to less than $25,000)"
                                          "Less than $35,000 ($25,000 to less than $35,000)",
                                          "Less than $50,000 ($35,000 to less than $50,000)",
                                          "Less than $75,000 ($50,000 to less than $75,000)"))

bp=st.sidebar.selectbox("Do you have blood pressure?",options=("No", "Yes"))

dep = st.sidebar.selectbox("Do You have a Depression?",options=("No", "Yes"))

chol = st.sidebar.selectbox("When you Check Cholestrol?",options=(
    "Have never had cholesterol checked",
    "Did not have cholesterol checked in past 5 years",
    "Had cholesterol checked in past 5 years",
))

heartDiease = st.sidebar.selectbox("Do you have Heart disease?", options=("No", "Yes"))

arthidis = st.sidebar.selectbox("Do you have arthidis?",options=("No", "Yes"))

dataToPredic = pd.DataFrame({
   "_SEX": [Gender],
   "_AGE80": [Age],
   "_BMI5CAT": [BMI],
   "SMOKE100": [Smoking],
   "GENHLTH": [genHealth],
   "EXERANY2": [physAct],
   "ASTHMA3": [asthma],
   "CHCKDNY2": [kidneyDisease],
   "INCOME3": [income],
   "BPHIGH6": [bp],
   "ADDEPEV3":[dep],
   "CHOLCHK3":[chol],
   "CVDCRHD4": [heartDiease],
   "ARTHDIS2":[arthidis]
 })

dataToPredic.replace("Yes",1,inplace=True)
dataToPredic.replace("No",2,inplace=True)

dataToPredic.replace("18-24",1,inplace=True)
dataToPredic.replace("25-29",2,inplace=True)
dataToPredic.replace("30-34",3,inplace=True)
dataToPredic.replace("35-39",4,inplace=True)
dataToPredic.replace("40-44",5,inplace=True)
dataToPredic.replace("45-49",6,inplace=True)
dataToPredic.replace("50-54",7,inplace=True)
dataToPredic.replace("55-59",8,inplace=True)
dataToPredic.replace("60-64",9,inplace=True)
dataToPredic.replace("65-69",10,inplace=True)
dataToPredic.replace("70-74",11,inplace=True)
dataToPredic.replace("75-79",12,inplace=True)
dataToPredic.replace("80 or older",13,inplace=True)

dataToPredic.replace("$75,000 or more",1,inplace=True)
dataToPredic.replace("Less than $10,000",2,inplace=True)
dataToPredic.replace("Less than $15,000 ($10,000 to less than $15,000)",3,inplace=True)
dataToPredic.replace("Less than $20,000 ($15,000 to less than $20,000)",4,inplace=True)
dataToPredic.replace("Less than $25,000 ($20,000 to less than $25,000)",5,inplace=True)
dataToPredic.replace("Less than $35,000 ($25,000 to less than $35,000)",6,inplace=True)
dataToPredic.replace("Less than $50,000 ($35,000 to less than $50,000)",7,inplace=True)
dataToPredic.replace("Less than $75,000 ($50,000 to less than $75,000)",8,inplace=True)

dataToPredic.replace("Female",1,inplace=True)
dataToPredic.replace("Male",2,inplace=True)


dataToPredic.replace("Excellent",1,inplace=True)
dataToPredic.replace("Very good",2,inplace=True)
dataToPredic.replace("Good",3,inplace=True)
dataToPredic.replace("Fair",4,inplace=True)
dataToPredic.replace("Poor",5,inplace=True)


dataToPredic.replace("Have never had cholesterol checked",1,inplace=True)
dataToPredic.replace("Did not have cholesterol checked in past 5 years",2,inplace=True)
dataToPredic.replace("Had cholesterol checked in past 5 years",3,inplace=True)

@st.cache_resource
def loadModels():
    # Load the models dictionary
    filename = 'model/Diabetic-Risk-Prediction-models.pkl'
    return joblib.load(open(filename, 'rb'))

if submit:

    loaded_models = loadModels()
    # Access each model

    dataset=pd.read_csv("data/diabetic_prediction.csv")
    
    logistic_model = loaded_models['LogisticRegression']
    random_forest_model = loaded_models['RandomForest']
    X_train = loaded_models['X_Train']  # Remove the comma
    x_test = loaded_models['X_Test']    # Remove the comma
    y_train = loaded_models['Y_Train']  # Remove the comma
    y_test = loaded_models['Y_Test']    # Remove the comma

    st.write(loaded_models)
    
    st.write(len(dataset))
    st.write(len(x_test))
    st.write(len(y_test))

    logistic_Pred = logistic_model.predict(x_test)
    random_forest_Pred = random_forest_model.predict(x_test)

    confusionMatric = confusion_matrix(y_test, logistic_Pred)
    confusionMatricRF = confusion_matrix(y_test, random_forest_Pred)

    st.write("Confusion Matrix")
    st.write(confusionMatric)
    st.write("Confusion Matrix for Random Forest")
    st.write(confusionMatricRF)


    # Predict using both models
    logistic_result = logistic_model.predict(dataToPredic)
    logistic_result_prob = logistic_model.predict_proba(dataToPredic)
    logistic_result_prob1 = round(logistic_result_prob[0][1] * 100, 2)

    random_forest_result = random_forest_model.predict(dataToPredic)
    random_forest_result_prob = random_forest_model.predict_proba(dataToPredic)
    random_forest_result_prob1 = round(random_forest_result_prob[0][1] * 100, 2)
    

    st.write("Your Diabetic Risk Predictions are ready!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Log. Regression Prediction")
        if logistic_result_prob1 < 30:
            st.markdown(f"**The probability that you'll have diabetic risk is {round(logistic_result_prob[0][1] * 100, 2)}%. You are healthy!**")
            st.image("images/heart-okay.jpg", caption="Your heart seems to be okay! - Dr. ByteForza Regression")
        else:
            st.markdown(f"**The probability that you'll have diabetic risk is {round(logistic_result_prob[0][1] * 100, 2)}%. You might be at risk!**")
            st.image("images/heart-bad.jpg", caption="Your heart might be at risk! - Dr. ByteForza Regression")
    with col2:
        st.subheader("Random Forest Prediction")
        if random_forest_result_prob1 < 30:
            st.markdown(f"**The probability that you'll have diabetic risk is {round(random_forest_result_prob[0][1] * 100, 2)}%. You are healthy!**")
            st.image("images/heart-okay.jpg", caption="Your heart seems to be okay! - Dr. ByteForza Forest")
        else:
            st.markdown(f"**The probability that you'll have diabetic risk is {round(random_forest_result_prob[0][1] * 100, 2)}%. You might be at risk!**")
            st.image("images/heart-bad.jpg", caption="Your heart might be at risk! - Dr. ByteForza Forest")

  
#test commit from vscode to dev branch
  
