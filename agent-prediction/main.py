from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import List
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class PredictionInput(BaseModel):
    city: str
    product: str

class PredictionOutput(BaseModel):
    agent: str

app = FastAPI()

# Load your trained model
model = joblib.load('model.joblib')
le_city = joblib.load('le_city.joblib')
le_product = joblib.load('le_product.joblib')
le_agent = joblib.load('le_agent.joblib')



# Define your API endpoint
@app.post('/predict', response_model=List[PredictionOutput])
def predict(input: dict):
    print('Input data:', input)
    # Encode your categorical features using LabelEncoder
    input_encoded = pd.DataFrame(input,index=[0])
    input_encoded['city'] = le_city.transform(input_encoded['city'])
    input_encoded['product'] = le_product.transform(input_encoded['product'])
    
    # Make a prediction using your trained model
    predictions = model.predict(input_encoded[['city', 'product']])
    predictions = le_agent.inverse_transform(predictions)
    
    # Return the predictions as a JSON response
    return [{'agent': agent} for agent in predictions]
