from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import numpy as np
import traceback

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Spending Prediction API",
    description="Predict spending for 7, 14, 30 days (category-wise)",
    version="1.0"
)

# =========================
# REQUEST MODELS
# =========================
class Transaction(BaseModel):
    date: str          # "YYYY-MM-DD"
    amount: float
    category: str

class PredictRequest(BaseModel):
    transactions: List[Transaction]
    days: int  # Days to predict (7, 14, or 30)

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok"}

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not req.transactions:
            raise HTTPException(status_code=400, detail="No transactions provided")

        # -------------------------
        # Convert to DataFrame
        # -------------------------
        df = pd.DataFrame([t.dict() for t in req.transactions])

        # Parse dates
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

        # Only expenses (treat all as positive spending)
        df["amount"] = df["amount"].abs()

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid expense data")

        # -------------------------
        # DAILY AGGREGATION
        # -------------------------
        daily = df.groupby(["date", "category"])["amount"].sum().reset_index()

        if len(daily) < 5:
            raise HTTPException(
                status_code=400,
                detail="Not enough data (minimum 5 days required)"
            )

        # -------------------------
        # CATEGORY ENCODING
        # -------------------------
        le = LabelEncoder()
        daily["category_id"] = le.fit_transform(daily["category"])

        # -------------------------
        # Calculate category statistics
        # -------------------------
        category_stats = {}
        for cat_id in daily["category_id"].unique():
            cat_data = daily[daily["category_id"] == cat_id]["amount"]
            category_stats[cat_id] = {
                'mean': float(cat_data.mean()),
                'std': float(cat_data.std()),
                'median': float(cat_data.median()),
                'max': float(cat_data.max())
            }

        # -------------------------
        # FEATURE ENGINEERING
        # -------------------------
        daily = daily.sort_values(["category_id", "date"]).reset_index(drop=True)
        daily["dow"] = daily["date"].dt.dayofweek
        daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
        daily["day"] = daily["date"].dt.day
        daily["month"] = daily["date"].dt.month

        # LAG FEATURES using shift and rolling
        daily["lag_1"] = daily.groupby("category_id")["amount"].shift(1)
        
        daily["lag_7"] = daily.groupby("category_id")["amount"].transform(
            lambda x: x.rolling(7, min_periods=1).mean().shift(1)
        )
        
        daily["lag_14"] = daily.groupby("category_id")["amount"].transform(
            lambda x: x.rolling(14, min_periods=1).mean().shift(1)
        )

        # Fill NaN with category mean
        for col in ["lag_1", "lag_7", "lag_14"]:
            for cat_id in daily["category_id"].unique():
                mask = daily["category_id"] == cat_id
                cat_mean = daily.loc[mask, "amount"].mean()
                daily.loc[mask, col] = daily.loc[mask, col].fillna(cat_mean)
        
        daily.fillna(0, inplace=True)

        FEATURES = [
            "category_id",
            "dow",
            "is_weekend",
            "day",
            "month",
            "lag_1",
            "lag_7",
            "lag_14"
        ]

        # -------------------------
        # TRAIN MODEL (with strong regularization)
        # -------------------------
        X = daily[FEATURES]
        y = daily["amount"]

        model = GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.02,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )
        model.fit(X, y)

        # -------------------------
        # PREDICTION WITH BETTER STABILITY
        # -------------------------
        last_date = daily["date"].max()
        results = []

        for cat_id in daily["category_id"].unique():
            # Get historical data for this category
            cat_history = daily[daily["category_id"] == cat_id].copy()
            stats = category_stats[cat_id]
            
            # Use historical mean as baseline - this prevents explosion
            baseline_daily = stats['mean']
            max_prediction = stats['mean'] + 1.5 * stats['std']
            
            # If std is very small or NaN, cap at 1.5x mean
            if pd.isna(max_prediction) or max_prediction < baseline_daily:
                max_prediction = baseline_daily * 1.5

            # For predictions, use ONLY historical values for lag features
            # This prevents the feedback loop that causes explosion
            for i in range(req.days):
                next_date = last_date + timedelta(days=i + 1)

                # CRITICAL: Use only actual historical data for lags, not predictions
                historical_amounts = cat_history["amount"].values
                
                row = {
                    "category_id": cat_id,
                    "dow": next_date.weekday(),
                    "is_weekend": int(next_date.weekday() >= 5),
                    "day": next_date.day,
                    "month": next_date.month,
                    # Use historical averages, not predicted values
                    "lag_1": baseline_daily,  # Use mean instead of last prediction
                    "lag_7": historical_amounts[-7:].mean() if len(historical_amounts) >= 7 else baseline_daily,
                    "lag_14": historical_amounts[-14:].mean() if len(historical_amounts) >= 14 else baseline_daily,
                }

                # Predict
                pred = model.predict(pd.DataFrame([row])[FEATURES])[0]
                
                # Apply multiple safety caps
                pred = max(0, pred)  # No negative
                pred = min(pred, max_prediction)  # Cap at mean + 1.5*std
                pred = min(pred, stats['max'] * 1.2)  # Never exceed 120% of historical max
                
                # Blend with historical mean (80% prediction, 20% historical mean)
                # This stabilizes predictions
                pred = 0.8 * pred + 0.2 * baseline_daily
                
                # Apply decay for far future predictions
                decay = 0.99 ** i
                pred = pred * decay

                results.append({
                    "date": str(next_date.date()),
                    "category": le.inverse_transform([cat_id])[0],
                    "predicted_amount": round(pred, 2)
                })

        forecast = pd.DataFrame(results)

        # -------------------------
        # BUILD RESPONSE IN FRONTEND FORMAT
        # -------------------------
        requested_days = req.days
        
        # Calculate total per category for requested days
        category_totals = forecast.groupby("category")["predicted_amount"].sum().round(2).to_dict()
        
        # Format predictions as array of objects
        predictions_array = []
        for category, amount in category_totals.items():
            predictions_array.append({
                "category": category,
                "amount": float(amount),
                "days": requested_days,
                "isDemo": False
            })
        
        # Return the exact format frontend expects
        return {
            "predictions": predictions_array,
            "success": True,
            "days": requested_days
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        print(f"Error occurred: {error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )
