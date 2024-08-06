from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the trained model, label encoders, scaler, and feature names
model = joblib.load('random_forest_regressor.pkl')
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = joblib.load(file)
scaler = joblib.load('scaler.pkl')
with open('feature_names.pkl', 'rb') as file:
    feature_names = joblib.load(file)

class ContributionInput(BaseModel):
    amountPaid: float
    region: str
    occupation: str

class ContributionOutput(BaseModel):
    predicted_contribution: float

# Comprehensive mappings for region and occupation
region_mapping = {
    "Mombasa": 0, "Kwale": 1, "Kilifi": 2, "Tana River": 3, "Lamu": 4, "Taita-Taveta": 5,
    "Garissa": 6, "Wajir": 7, "Mandera": 8, "Marsabit": 9, "Isiolo": 10, "Meru": 11,
    "Tharaka-Nithi": 12, "Embu": 13, "Kitui": 14, "Machakos": 15, "Makueni": 16,
    "Nyandarua": 17, "Nyeri": 18, "Kirinyaga": 19, "Murang'a": 20, "Kiambu": 21,
    "Turkana": 22, "West Pokot": 23, "Samburu": 24, "Trans Nzoia": 25, "Uasin Gishu": 26,
    "Elgeyo-Marakwet": 27, "Nandi": 28, "Baringo": 29, "Laikipia": 30, "Nakuru": 31,
    "Narok": 32, "Kajiado": 33, "Kericho": 34, "Bomet": 35, "Kakamega": 36, "Vihiga": 37,
    "Bungoma": 38, "Busia": 39, "Siaya": 40, "Kisumu": 41, "Homa Bay": 42, "Migori": 43,
    "Kisii": 44, "Nyamira": 45, "Nairobi": 46
}

occupation_mapping = {
    "Managerial": 0, "Professional": 1, "Technical": 2, "Clerical": 3, "Service": 4,
    "Skilled Manual": 5, "Semi-Skilled Manual": 6, "Unskilled Manual": 7, "Agricultural": 8, "Other": 9
}

@app.get("/")
def read_root():
    return FileResponse("templates/index.html")

@app.post("/predict", response_model=ContributionOutput)
def predict_contribution(input_data: ContributionInput):
    try:
        # Log the input data
        logger.info(f"Received input data: {input_data}")

        # Prepare the input data for prediction
        data = {
            'how much paid in last month.1': [input_data.amountPaid],
            'region': [input_data.region],
            'occupation (grouped)': [input_data.occupation]
        }
        
        # Convert the input data to a DataFrame
        df = pd.DataFrame(data)
        logger.info(f"DataFrame after creation: {df}")
        
        # Map the user input values to the encoded values expected by the model
        df['region'] = df['region'].map(region_mapping)
        df['occupation (grouped)'] = df['occupation (grouped)'].map(occupation_mapping)
        logger.info(f"DataFrame after mapping: {df}")
        
        # Check if the mappings were successful
        if df['region'].isnull().any() or df['occupation (grouped)'].isnull().any():
            logger.error("Invalid region or occupation input value.")
            raise ValueError("Invalid region or occupation input value.")
        
        # Ensure the numeric data is of type float
        df['how much paid in last month.1'] = df['how much paid in last month.1'].astype(float)
        logger.info(f"DataFrame after type conversion: {df}")

        # Ensure all required features are present and in the correct order
        df = df[feature_names]
        logger.info(f"DataFrame after reindexing: {df}")
        
        # Standardize the data using the previously fitted scaler
        df_rescaled = scaler.transform(df)
        logger.info(f"DataFrame after scaling: {df_rescaled}")
        
        # Predict the contribution amount
        prediction = model.predict(df_rescaled)
        
        # Extract the predicted contribution amount
        predicted_contribution = prediction[0]
        logger.info(f"Predicted contribution: {predicted_contribution}")
        
        return ContributionOutput(predicted_contribution=predicted_contribution)
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
