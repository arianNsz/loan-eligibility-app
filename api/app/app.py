import streamlit as st
import numpy as np
from joblib import load


### The UI
st.title("Loan Approval Predictor")

st.markdown("### Disclaimer:")
st.markdown("This is not a financial product or adivsory tool!" )
st.markdown("This web app is part of my professional portfolio. You can find the tutorial for building this app on my Medium posts [here](https://medium.com/@arian.naseh). Please note the purpose of the tutorial was not how to train an accurate model, but how to deploy the model. So the model is not validated." )
st.markdown("")
st.markdown("And please feel free to check out my personal website [here](https://ariannsz.github.io/).")
st.markdown("")
st.markdown("### Enter the details and click on the button to check if you are eligible for loan or not!")
dependents = st.number_input("Number of dependents", value=0, step=1, min_value=0, max_value=5)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
education = 1 if education == "Graduate" else 0
self_emp = st.selectbox("Self Employed", ["Yes", "No"])
self_emp = 1 if self_emp == "Yes" else 0
income = st.number_input("Income per annum", value=2000, step=1000, min_value=2000, max_value=100000)
loan = st.number_input("Loan amount", value=1000, step=1000, min_value=1000, max_value=395000)
term = st.number_input("Loan term", value=1, step=1, min_value=1, max_value=30)
credit = st.number_input("Credit Score", value=300, step=1, min_value=300, max_value=900)
res_val = st.number_input("Value of residential assets", value=0, step=1000, min_value=-1000, max_value=291000)
com_val = st.number_input("Value of commercial assets", value=0, step=1000, min_value=0, max_value=194000)
lux_val = st.number_input("Value of luxury assets", value=0, step=1000, min_value=0, max_value=392000)
bank_val = st.number_input("Value of bank assets", value=0, step=1000, min_value=0, max_value=147000)
button = st.button("Check Eligibility")


# Loading the sclaers
no_of_dependets_scaler = load('./models/no_of_dependents_scaler.joblib')
income_annum_scaler = load('./models/income_annum_scaler.joblib')
loan_amount_scaler = load('./models/loan_amount_scaler.joblib')
loan_term_scaler = load('./models/loan_term_scaler.joblib')
cibil_score_scaler = load('./models/cibil_score_scaler.joblib')
residential_assets_value_scaler = load('./models/residential_assets_value_scaler.joblib')
commercial_assets_value_scaler = load('./models/commercial_assets_value_scaler.joblib')
luxury_assets_value_scaler = load('./models/luxury_assets_value_scaler.joblib')
bank_asset_value_scaler = load('./models/bank_asset_value_scaler.joblib')

# Loading the classifier
xgb = load('./models/loan_classifier.joblib')

features = [dependents, education, self_emp, income, loan, term, credit, res_val, com_val, lux_val, bank_val]

def predict(features: list):
    """
    Predicts the loan approval based on the given features.

    Parameters:
    features (list): A list of features for the loan application.

    Returns:
    prediction: The predicted loan approval.
    """
    # Convert the data into a numpy array
    features = np.array(features).reshape(1, -1)
    # Scale the data
    features[:, 0] = no_of_dependets_scaler.transform(features[:, 0].reshape(-1, 1)).reshape(1, -1)
    features[:, 3] = income_annum_scaler.transform(features[:, 3].reshape(-1, 1)).reshape(1, -1)
    features[:, 4] = loan_amount_scaler.transform(features[:, 4].reshape(-1, 1)).reshape(1, -1)
    features[:, 5] = loan_term_scaler.transform(features[:, 5].reshape(-1, 1)).reshape(1, -1)
    features[:, 6] = cibil_score_scaler.transform(features[:, 6].reshape(-1, 1)).reshape(1, -1)
    features[:, 7] = residential_assets_value_scaler.transform(np.log1p(features[:, 7]).reshape(-1, 1)).reshape(1, -1)
    features[:, 8] = commercial_assets_value_scaler.transform(np.log1p(features[:, 8]).reshape(-1, 1)).reshape(1, -1)
    features[:, 9] = luxury_assets_value_scaler.transform(np.log1p(features[:, 9]).reshape(-1, 1)).reshape(1, -1)
    features[:, 10] = bank_asset_value_scaler.transform(np.log1p(features[:, 10]).reshape(-1, 1)).reshape(1, -1)

    prediction = xgb.predict(features)
    return prediction

# when the button is clicked fetch the response
if button:
    response = predict(features)
    print(response)
    if response==0:
        st.markdown("## Sorry!")
        st.write("You probably won't get the loan! :(")
    else:
        st.markdown("## Congratulations!")
        st.write("You most likely will get the loan! :D")
    
