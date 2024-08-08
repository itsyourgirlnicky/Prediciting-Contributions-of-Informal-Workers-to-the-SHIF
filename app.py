from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import joblib
import numpy as np
from google.oauth2 import service_account
from google.cloud import dialogflow_v2 as dialogflow
import requests

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

# Load the pre-trained model
model = joblib.load("random_forest_regressor.pkl")

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

        # Map the region and occupation to their corresponding numerical values
        region_encoded = region_mapping[input_data.region]
        occupation_encoded = occupation_mapping[input_data.occupation]

        # Prepare the input for the model
        model_input = np.array([[input_data.amountPaid, region_encoded, occupation_encoded]])
        logger.info(f"Model input: {model_input}")

        # Predict the contribution using the pre-trained model
        predicted_contribution = model.predict(model_input)[0]
        logger.info(f"Predicted contribution: {predicted_contribution}")

        return ContributionOutput(predicted_contribution=predicted_contribution)
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Dialogflow webhook endpoint
credentials = service_account.Credentials.from_service_account_file("shif-prediction-chatbot-59fe29b269e9.json")
project_id = "shif-prediction-chatbot"
search_engine_id = "96d81e96796cd4218"
api_key = "AIzaSyCXsk4Gt9GIZnC5c3rzBF7wfuyzQN6wqLM"

def search_internet(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={search_engine_id}&key={api_key}"
    logger.info(f"Searching internet with URL: {url}")
    response = requests.get(url)
    data = response.json()
    logger.info(f"Received data: {data}")
    if "items" in data:
        snippet = data["items"][0]["snippet"]
        logger.info(f"Search result: {snippet}")
        return snippet
    else:
        logger.info("No relevant information found.")
        return "No relevant information found."

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    try:
        req = await request.json()
        logger.info(f"Received Dialogflow request: {req}")
        session_id = req['session'].split('/')[-1]
        query_text = req['queryResult']['queryText']
        response_text = search_internet(query_text)

        return JSONResponse({
            "fulfillmentText": response_text
        })
    except Exception as e:
        logger.error(f"Exception occurred in webhook: {str(e)}")
        return JSONResponse({
            "fulfillmentText": f"Error occurred: {str(e)}"
        }, status_code=500)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
