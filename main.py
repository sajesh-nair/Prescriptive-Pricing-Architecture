import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input structure (Now includes model_type)
class MarketInputs(BaseModel):
    state: int
    zone: int
    category: int
    brand_type: int
    customer_gender: int
    customer_age: int
    base_price: float
    discount_percent: float
    sales_event: int
    competition_intensity: int
    inventory_pressure: int
    order_year: int
    order_month: int
    model_type: str = "xgboost" # Default routing

# Global artifacts
scaler = None
rf_model = None
xgb_model = None

@app.on_event("startup")
def load_artifacts():
    global scaler, rf_model, xgb_model
    try:
        scaler = joblib.load("scaler.pkl")
        rf_model = joblib.load("random_forest_regressor.pkl")
        if os.path.exists("xgboost_regressor.pkl"):
            xgb_model = joblib.load("xgboost_regressor.pkl")
        print("✅ ML Artifacts loaded into memory.")
    except Exception as e:
        print(f"⚠️ Warning during artifact load: {e}")

@app.get("/features")
def get_features():
    df = pd.read_csv("./data/indian_ecommerce_pricing_revenue_growth.csv")
    
    # 1. Print the exact columns to your Uvicorn terminal for debugging
    print("📌 EXACT CSV COLUMNS:", df.columns.tolist())
    
    # 2. Try to find the exact column names (handling potential lowercase variations)
    # If your columns are named something completely different (like 'Location' or 'Product_Line'), 
    # replace the strings inside the brackets below with your exact column names from the print statement!
    
    state_col = 'State' if 'State' in df.columns else 'state'
    category_col = 'Category' if 'Category' in df.columns else 'category'
    
    try:
        return {
            "states": sorted(df[state_col].dropna().unique().tolist()),
            "categories": sorted(df[category_col].dropna().unique().tolist()),
        }
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Column not found in CSV. Please check the terminal for the exact column names and update main.py. Error: {e}")

@app.post("/predict")
def get_prediction(inputs: MarketInputs):
    global scaler, rf_model, xgb_model

    # Route to the selected algorithm
    active_model = xgb_model if inputs.model_type == "xgboost" else rf_model
    
    if active_model is None:
        raise HTTPException(status_code=503, detail=f"{inputs.model_type} artifact not found on server.")

    # Convert incoming payload, drop model_type for inference
    data_dict = inputs.model_dump()
    data_dict.pop("model_type", None)
    
    df_input = pd.DataFrame([data_dict])
    
    # Scale and predict
    scaled_features = scaler.transform(df_input)
    prediction = active_model.predict(scaled_features)[0]
    
    return {"predicted_revenue": float(prediction)}