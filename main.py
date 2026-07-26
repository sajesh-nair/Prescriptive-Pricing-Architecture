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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global cache for serverless functions
_artifacts = {
    "scaler": None,
    "rf_model": None,
    "xgb_model": None
}

def load_artifact(filename):
    """Helper to safely load joblib artifacts in Vercel serverless context."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        # Fallback check directly in root directory
        path = filename
    
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    else:
        print(f"File not found: {filename} at path: {path}")
        return None

def get_scaler():
    if _artifacts["scaler"] is None:
        _artifacts["scaler"] = load_artifact("scaler.pkl")
    return _artifacts["scaler"]

def get_model(model_type: str):
    m_type = model_type.lower()
    if "xgb" in m_type or "xgboost" in m_type:
        if _artifacts["xgb_model"] is None:
            _artifacts["xgb_model"] = load_artifact("xgboost_regressor.pkl")
        return _artifacts["xgb_model"], "xgboost_regressor.pkl"
    else:
        if _artifacts["rf_model"] is None:
            _artifacts["rf_model"] = load_artifact("random_forest_regressor.pkl")
        return _artifacts["rf_model"], "random_forest_regressor.pkl"

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
    model_type: str = "xgboost"

@app.get("/features")
def get_features():
    csv_path = os.path.join(BASE_DIR, "data", "indian_ecommerce_pricing_revenue_growth.csv")
    if not os.path.exists(csv_path):
        csv_path = "./data/indian_ecommerce_pricing_revenue_growth.csv"

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=500, detail=f"CSV file missing at {csv_path}")

    df = pd.read_csv(csv_path)
    state_col = 'State' if 'State' in df.columns else 'state'
    category_col = 'Category' if 'Category' in df.columns else 'category'

    return {
        "states": sorted(df[state_col].dropna().unique().tolist()),
        "categories": sorted(df[category_col].dropna().unique().tolist()),
    }

@app.post("/predict")
def get_prediction(inputs: MarketInputs):
    scaler = get_scaler()
    if scaler is None:
        raise HTTPException(status_code=503, detail="Pipeline failure. Ensure scaler.pkl is uploaded.")

    active_model, filename = get_model(inputs.model_type)
    if active_model is None:
        raise HTTPException(status_code=503, detail=f"Pipeline failure. Ensure {filename} is generated.")

    data_dict = inputs.model_dump() if hasattr(inputs, "model_dump") else inputs.dict()
    data_dict.pop("model_type", None)

    df_input = pd.DataFrame([data_dict])
    scaled_features = scaler.transform(df_input)
    prediction = active_model.predict(scaled_features)[0]

    return {"predicted_revenue": float(prediction)}