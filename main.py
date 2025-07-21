import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

podium_model = joblib.load("model_podium.pkl")
race_model = joblib.load("model_racetime.pkl")
model_columns = joblib.load("model_columns.pkl")

app = FastAPI()

class PredictionInput(BaseModel):
    DriverCode: str
    team: str
    QualTime: float
    grid_position: int

@app.get("/")
def root():
    return {"message": "F1 Prediction API is running"}

@app.post("/predict")
def predict(input: PredictionInput):
    try:
        data = {
            'QualTime': input.QualTime,
            'grid_position': input.grid_position,
            'DriverCode': input.DriverCode,
            'team': input.team
        }
        df = pd.DataFrame([data])

        # One-hot encoding
        encoded_df = pd.get_dummies(df[['DriverCode', 'team']])
        numerical_df = df[['QualTime', 'grid_position']]
        combined_df = pd.concat([numerical_df, encoded_df], axis=1)

        # Align with model columns
        combined_df = combined_df.reindex(columns=model_columns, fill_value=0)

        # Predictions
        race_time_sec = race_model.predict(combined_df)[0]
        podium_prob = podium_model.predict_proba(combined_df)[0][1]

        # Format race time
        def format_time(seconds):
            if pd.isna(seconds): return "N/A"
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        formatted_time = format_time(race_time_sec)

        return {
            "Driver": input.DriverCode,
            "Team": input.team,
            "Grid": input.grid_position,
            "QualTime": input.QualTime,
            "PredictedRaceTime": formatted_time,
            "PodiumProbability": round(podium_prob, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
