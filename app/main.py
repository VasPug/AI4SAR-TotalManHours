from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()




class ValuesIn(BaseModel):
    subject_category: str
    subject_activity: str
    weather: str
    age: int

class PredictionOut(BaseModel):
    totalManHours: int


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: ValuesIn):
    totalManHours = predict_pipeline(payload.subject_category, payload.subject_activity, payload.weather, payload.age)
    return {"totalManHours": totalManHours}