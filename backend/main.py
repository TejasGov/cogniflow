import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import xgboost
import numpy as np
import pandas as pd
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Add CORS middleware for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cogniflow_xgboost.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# Initialize Groq client — set GROQ_API_KEY in your environment
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)

class TaskRequest(BaseModel):
    task_description: str

class PredictRequest(BaseModel):
    complexity_score: float
    estimated_steps: float
    priority_encoded: float
    avg_gaze: float
    avg_head_pose: float
    avg_eye_openness: float
    gaze_variance: float

@app.post("/api/analyze-task")
async def analyze_task(request: TaskRequest):
    prompt = f"""
You are a task analysis assistant. Analyze the following task and provide three metrics and a step breakdown:
1. "complexity": a score from 1 to 5 evaluating how complex the task is.
2. "steps": an integer estimating how many sub-steps the task will take.
3. "priority": an integer from 1 to 5 evaluating how high priority the task is.
4. "steps_list": an array of strings, where each string is a concise step to complete the task.

Task: {request.task_description}

Return ONLY a valid JSON object matching this schema exactly: {{"complexity": 0, "steps": 0, "priority": 0, "steps_list": ["step 1", "step 2"]}}.
"""
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        # Parse the JSON
        parsed = json.loads(content)
        return {
            "complexity": parsed.get("complexity", 3),
            "steps": parsed.get("steps", 10),
            "priority": parsed.get("priority", 3),
            "steps_list": parsed.get("steps_list", [])
        }
    except Exception as e:
        print(f"Error during Groq request: {e}")
        # Fallback values if the LLM fails
        return {"complexity": 3, "steps": 10, "priority": 3, "steps_list": []}

@app.post("/api/predict")
async def predict_risk(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="XGBoost model not loaded")
    
    # Create a DataFrame matching the model's expected features
    features = pd.DataFrame([{
        'complexity_score': request.complexity_score,
        'estimated_steps': request.estimated_steps,
        'priority_encoded': request.priority_encoded,
        'avg_gaze': request.avg_gaze,
        'avg_head_pose': request.avg_head_pose,
        'avg_eye_openness': request.avg_eye_openness,
        'gaze_variance': request.gaze_variance
    }])
    
    try:
        # predict_proba returns [[prob_class_0, prob_class_1]]
        # We assume class 1 is "Abandonment Risk"
        proba = model.predict_proba(features)[0]
        risk_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        return {"riskScore": risk_score}
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
