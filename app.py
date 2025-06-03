from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
async def predict(data: dict):
    prediction = model.predict([data["features"]])
    return {"prediction": prediction.tolist()}