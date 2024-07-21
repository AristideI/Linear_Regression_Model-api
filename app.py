from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from dataEncoder import convertRequestToModelValue

# Load the trained model
model = joblib.load("dementia.joblib")


class InputData(BaseModel):
    diabetic: int
    alcoholLevel: float
    heartRate: float
    bloodOxygenLevel: float
    bodyTemperature: float
    weight: float
    mriDelay: float
    age: float
    educationLevel: str
    dominantHand: str
    Gender: str
    Smoking_Status: str
    APOE_ε4: str
    Physical_Activity: str
    Depression_Status: str
    Cognitive_Test_Scores: float
    Medication_History: str
    Nutrition_Diet: str
    Sleep_Quality: str
    Chronic_Health_Conditions: str


# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the input data schema
class InputData(BaseModel):
    diabetic: int
    alcoholLevel: float
    heartRate: float
    bloodOxygenLevel: float
    bodyTemperature: float
    weight: float
    mriDelay: float
    age: float
    educationLevel: str
    dominantHand: str
    Gender: str
    Smoking_Status: str
    APOE_ε4: str
    Physical_Activity: str
    Depression_Status: str
    Cognitive_Test_Scores: float
    Medication_History: str
    Nutrition_Diet: str
    Sleep_Quality: str
    Chronic_Health_Conditions: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Dementia Prediction API"}


@app.post("/predict")
def predict(data: InputData):
    # Convert input data to the required format for the model
    input_data = [convertRequestToModelValue(data.dict())]

    # Make prediction
    prediction = model.predict(input_data)
    print(prediction[0])

    # Return the prediction as a response
    return {"prediction": round(prediction[0], 0)}
