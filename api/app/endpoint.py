from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from typing import List
from joblib import load
import uvicorn

# Load the classifier
xgb = load('./models/kloan_classifier.joblib')

# loading the sclaers
no_of_dependets_scaler = load('./models/kno_of_dependents_scaler.joblib')
income_annum_scaler = load('./models/kincome_annum_scaler.joblib')
loan_amount_scaler = load('./models/kloan_amount_scaler.joblib')
loan_term_scaler = load('./models/kloan_term_scaler.joblib')
cibil_score_scaler = load('./models/kcibil_score_scaler.joblib')
residential_assets_value_scaler = load('./models/kresidential_assets_value_scaler.joblib')
commercial_assets_value_scaler = load('./models/kcommercial_assets_value_scaler.joblib')
luxury_assets_value_scaler = load('./models/kluxury_assets_value_scaler.joblib')
bank_asset_value_scaler = load('./models/kbank_asset_value_scaler.joblib')

app = FastAPI()

# class Data(BaseModel):
#     # 0 no_of_dependets, 1 education, 2 self_employed, 3 income_annum, 4 loan_amount, 
#     # 5 loan_term, 6 cibil_score, 7 residential_assets_value, 
#     # 8 commercial_assets_value, 9 luxury_assets_value, 10 bank_asset_value
#     features: List[float] = Field(..., min_items=11, max_items=11)
class Data(BaseModel):
    no_of_dependets: int
    education: int
    self_employed: int
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int

@app.post('/predict')
def predict(data: Data):
    # Convert the data into a numpy array
    features = np.array(data.features).reshape(1, -1)
    print(features)
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
    # Make a prediction
    print(features)

    prediction = xgb.predict(features)


    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)