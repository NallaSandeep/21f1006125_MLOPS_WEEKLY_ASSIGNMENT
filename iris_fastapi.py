from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Load model once when the application starts
MODEL_PATH = "/app/model"

model = mlflow.pyfunc.load_model(MODEL_PATH)

app = FastAPI(title="Iris Prediction API")


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: IrisRequest):
    input_df = pd.DataFrame([request.model_dump()])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_class": prediction
    }