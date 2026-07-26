import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve the directory where main.py resides to handle Vercel deployment paths properly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the expected input structure
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
    model_type: str = "xgboost"  # Default routing: "xgboost" or "random_forest"

# Global artifacts
scaler = None
rf_model = None
xgb_model = None

@app.on_event("startup")
def load_artifacts():
    global scaler, rf_model, xgb_model
    try:
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
        rf_path = os.path.join(BASE_DIR, "random_forest_regressor.pkl")
        xgb_path = os.path.join(BASE_DIR, "xgboost_regressor.pkl")

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            print(f"❌ Missing scaler file: {scaler_path}")

        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
        else:
            print(f"❌ Missing Random Forest model file: {rf_path}")

        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
        else:
            print(f"❌ Missing XGBoost model file: {xgb_path}")

        print("✅ ML Artifacts startup loading routine completed.")
    except Exception as e:
        print(f"⚠️ Warning during artifact load: {e}")

@app.get("/features")
def get_features():
    csv_path = os.path.join(BASE_DIR, "data", "indian_ecommerce_pricing_revenue_growth.csv")
    
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=500, 
            detail=f"CSV file not found on server at path: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    
    # Debug print for server logs
    print("📌 EXACT CSV COLUMNS:", df.columns.tolist())
    
    state_col = 'State' if 'State' in df.columns else 'state'
    category_col = 'Category' if 'Category' in df.columns else 'category'
    
    try:
        return {
            "states": sorted(df[state_col].dropna().unique().tolist()),
            "categories": sorted(df[category_col].dropna().unique().tolist()),
        }
    except KeyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Column not found in CSV. Error: {e}"
        )

@app.post("/predict")
def get_prediction(inputs: MarketInputs):
    global scaler, rf_model, xgb_model

    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler artifact not loaded on server.")

    # Select requested model
    active_model = xgb_model if inputs.model_type.lower() == "xgboost" else rf_model
    
    if active_model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Pipeline failure. Ensure {inputs.model_type} .pkl is generated and tracked in Git."
        )

    # Convert incoming payload (Pydantic V2 & V1 compatibility)
    data_dict = inputs.model_dump() if hasattr(inputs, "model_dump") else inputs.dict()
    data_dict.pop("model_type", None)
    
    df_input = pd.DataFrame([data_dict])
    
    # Scale and predict
    scaled_features = scaler.transform(df_input)
    prediction = active_model.predict(scaled_features)[0]
    
    return {"predicted_revenue": float(prediction)}