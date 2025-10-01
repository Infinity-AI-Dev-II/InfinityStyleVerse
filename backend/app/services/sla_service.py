from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import time
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import json

# Defining the input model
class PredictionInput(BaseModel):
    latency_ms: float
    queue_time_ms: float
    mean_latency_ms: float
    var_latency_ms: float

# Initializing the FastAPI app
app = FastAPI()

# Loading the trained model 
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "sla_model.pkl"
model = joblib.load(MODEL_PATH)

@app.post('/predict/sla_risk')
async def predict_sla_risk(input_data: PredictionInput):
    start_time = time.time()
    
    # Convert input to DataFrame
    data = {
        "latency_ms": input_data.latency_ms,
        "queue_time_ms": input_data.queue_time_ms,
        "mean_latency_ms": input_data.mean_latency_ms,
        "var_latency_ms": input_data.var_latency_ms
    }
    input_df = pd.DataFrame([data])

    # Predict risk score
    try:
        risk_score = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Load rules
    with open('sla_actions.json', 'r') as f:
        rules = json.load(f)
    action = next((band["action"] for band in rules["risk_bands"] if band["min_risk"] <= risk_score < band["max_risk"]), rules["default_action"])

    # Measure latency
    latency_ms = (time.time() - start_time) * 1000
    p95_target = 50

    # Return response
    response = {
        "risk_score": float(risk_score),
        "action": action,
        "latency_ms": float(latency_ms),
        "p95_target_met": latency_ms <= p95_target
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)