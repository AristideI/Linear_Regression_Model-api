from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load("studentperfomance.joblib")
scaler = joblib.load("scaler.joblib")
# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify a list of origins that are allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the input data schema
class InputData(BaseModel):
    HoursStudied: int
    PreviousScores: int
    ExtracurricularActivities: int
    SleepHours: int
    SampleQuestionPapersPracticed: int
    # Add all the necessary features here
    # featureN: float


# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Performance Prediction API"}


# Define a prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to the required format
    input_data = np.array(
        [
            [
                data.HoursStudied,
                data.PreviousScores,
                data.ExtracurricularActivities,
                data.SleepHours,
                data.SampleQuestionPapersPracticed,
            ]
        ]
    )
    # Scale the input data if necessary
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return the prediction as a response
    return {"prediction": prediction[0]}
