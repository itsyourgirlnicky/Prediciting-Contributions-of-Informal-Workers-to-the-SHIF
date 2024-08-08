from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import joblib
import numpy as np
import requests
from google.oauth2 import service_account
from google.cloud import dialogflow_v2 as dialogflow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ContributionInput(BaseModel):
    amountPaid: float
    region: str
    occupation: str

class ContributionOutput(BaseModel):
    predicted_contribution: float

# Load the pre-trained model, scaler, and label encoders
try:
    model = joblib.load("random_forest_regressor.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    with open("feature_names.pkl", 'rb') as file:
        feature_names = joblib.load(file)
    logger.info("Model and related files loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or related files: {e}")
    raise e

# Helper function to check if value exists in the encoder classes
def is_valid_label(value, encoder):
    return value in encoder.classes_

@app.get("/")
def read_root():
    return FileResponse("templates/landingpage.html")

@app.get("/index.html")
def read_index():
    return FileResponse("templates/index.html")

@app.post("/predict", response_model=ContributionOutput)
def predict_contribution(input_data: ContributionInput):
    try:
        # Log the input data
        logger.info(f"Received input data: {input_data}")

        # Check if the region and occupation are in the known lists
        if not is_valid_label(input_data.region, label_encoders['region']):
            raise ValueError(f"Region '{input_data.region}' is not recognized. Please provide a valid region.")
        if not is_valid_label(input_data.occupation, label_encoders['occupation (grouped)']):
            raise ValueError(f"Occupation '{input_data.occupation}' is not recognized. Please provide a valid occupation.")

        # Map the region and occupation to their corresponding numerical values
        region_encoded = label_encoders['region'].transform([input_data.region])[0]
        occupation_encoded = label_encoders['occupation (grouped)'].transform([input_data.occupation])[0]

        # Prepare the input for the model
        model_input = np.array([[input_data.amountPaid, region_encoded, occupation_encoded]])
        logger.info(f"Model input: {model_input}")

        # Scale the input
        model_input_scaled = scaler.transform(model_input)
        logger.info(f"Scaled model input: {model_input_scaled}")

        # Predict the contribution using the pre-trained model
        predicted_contribution = model.predict(model_input_scaled)[0]
        logger.info(f"Predicted contribution: {predicted_contribution}")

        # Calculate SHIF contribution based on 2.75% of the predicted contribution
        shif_contribution = predicted_contribution * 0.0275
        logger.info(f"SHIF contribution: {shif_contribution}")

        return ContributionOutput(predicted_contribution=shif_contribution)
    except ValueError as e:
        logger.error(f"ValueError occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Dialogflow webhook endpoint
try:
    credentials = service_account.Credentials.from_service_account_file("shif-prediction-chatbot-59fe29b269e9.json")
    project_id = "shif-prediction-chatbot"
    search_engine_id = "96d81e96796cd4218"
    api_key = "AIzaSyCXsk4Gt9GIZnC5c3rzBF7wfuyzQN6wqLM"
    logger.info("Google credentials loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Google credentials: {e}")
    raise e

def search_internet(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={search_engine_id}&key={api_key}"
        logger.info(f"Searching internet with URL: {url}")
        response = requests.get(url)
        data = response.json()
        logger.info(f"Received data: {data}")
        if "items" in data:
            return data["items"][0]["snippet"]
        else:
            return "No relevant information found."
    except Exception as e:
        logger.error(f"Error occurred while searching the internet: {e}")
        return "Error occurred while searching the internet."

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    try:
        req = await request.json()
        session_id = req['session'].split('/')[-1]
        query_text = req['queryResult']['queryText']
        response_text = search_internet(query_text)

        return JSONResponse({
            "fulfillmentText": response_text
        })
    except Exception as e:
        logger.error(f"Exception occurred in webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Webhook processing error")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
