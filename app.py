from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Load the trained model and label encoders
model = joblib.load('random_forest_model.pkl')
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = joblib.load(file)

class ContributionInput(BaseModel):
    amountPaid: float
    region: str
    occupation: str

class ContributionOutput(BaseModel):
    predicted_contribution: float

@app.get("/")
def read_root():
    return FileResponse("templates/index.html")

@app.post("/predict", response_model=ContributionOutput)
def predict_contribution(input_data: ContributionInput):
    try:
        # Prepare the input data for prediction
        data = {
            'how much paid in last month.1': [input_data.amountPaid],
            'region': [input_data.region],
            'occupation (grouped)': [input_data.occupation]
        }
        
        # Convert the input data to a DataFrame
        df = pd.DataFrame(data)
        
        # Transform categorical features using label encoders
        for col in ['region', 'occupation (grouped)']:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])
        
        # Ensure the numeric data is of type float
        df['how much paid in last month.1'] = df['how much paid in last month.1'].astype(float)
        
        # Standardize the data using the previously fitted scaler
        scaler = MinMaxScaler()
        df_rescaled = scaler.fit_transform(df)
        
        # Predict the contribution amount
        prediction = model.predict(df_rescaled)
        
        # Extract the predicted contribution amount
        predicted_contribution = prediction[0]
        
        return ContributionOutput(predicted_contribution=predicted_contribution)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
