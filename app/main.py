from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title="Diabetes Prediction API")

MODEL_PATH = "models/model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].max()
    return {"prediction": int(pred), "confidence": float(prob)}
