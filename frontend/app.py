import streamlit as st
import requests

st.title("Loan Approval Predictor")
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

features = [dependents, education, self_emp, income, loan, term, credit, res_val, com_val, lux_val, bank_val]

# when the button is clicked fetch the response
if button:
    response = requests.post("http://api:8000/predict", json={"features": features})
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        if prediction[0]==0:
            st.markdown("## Sorry!")
            st.write("You probably won't get the loan! :(")
        else:
            st.markdown("## Congratulations!")
            st.write("You most likely will get the loan! :D")
    else:
        st.markdown("## Something went wrong! ")
        st.write("Please try again later.")

