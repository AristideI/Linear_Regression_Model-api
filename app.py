from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load("dementia_model.joblib")

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

    # Make prediction
    prediction = model.predict(input_data)

    # Return the prediction as a response
    return {"prediction": prediction[0]}

# Run the FastAPI application
# uvicorn app:app --reload