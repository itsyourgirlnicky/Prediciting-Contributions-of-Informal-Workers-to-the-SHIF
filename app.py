from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

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

# Comprehensive mappings for region and occupation (if needed for other purposes)
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

        # Compute the contribution as 2.75% of the amountPaid
        predicted_contribution = input_data.amountPaid * 0.0275
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
