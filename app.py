from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import numpy as np
import traceback
from datetime import datetime

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
# HELPER FUNCTIONS
# =========================
def calculate_lag_features(group, days_back):
    """Calculate lag features for a specific group"""
    group = group.sort_values('date').copy()
    
    # Create lag columns
    for lag in [1, 7, 14]:
        if len(group) >= lag:
            group[f'lag_{lag}'] = group['amount'].shift(lag)
        else:
            # If not enough history, use mean
            group[f'lag_{lag}'] = group['amount'].mean()
    
    # Rolling averages
    if len(group) >= 7:
        group['rolling_7'] = group['amount'].rolling(7, min_periods=1).mean().shift(1)
    else:
        group['rolling_7'] = group['amount'].mean()
        
    if len(group) >= 14:
        group['rolling_14'] = group['amount'].rolling(14, min_periods=1).mean().shift(1)
    else:
        group['rolling_14'] = group['amount'].mean()
    
    return group

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not req.transactions:
            raise HTTPException(status_code=400, detail="No transactions provided")
        
        if req.days not in [7, 14, 30]:
            raise HTTPException(status_code=400, detail="Days must be 7, 14, or 30")

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

        # Ensure we have enough unique dates
        unique_dates = daily['date'].nunique()
        if unique_dates < 5:
            raise HTTPException(
                status_code=400,
                detail=f"Only {unique_dates} unique dates found. Need at least 5."
            )

        # -------------------------
        # CATEGORY ENCODING
        # -------------------------
        le = LabelEncoder()
        daily["category_id"] = le.fit_transform(daily["category"])
        
        # Get category mapping
        category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        # -------------------------
        # FEATURE ENGINEERING
        # -------------------------
        # Sort by date first
        daily = daily.sort_values("date").reset_index(drop=True)
        
        # Apply lag features per category
        daily_with_lags = daily.groupby('category_id').apply(
            lambda x: calculate_lag_features(x, req.days)
        ).reset_index(drop=True)
        
        # Create time features
        daily_with_lags["dow"] = daily_with_lags["date"].dt.dayofweek
        daily_with_lags["is_weekend"] = (daily_with_lags["dow"] >= 5).astype(int)
        daily_with_lags["day"] = daily_with_lags["date"].dt.day
        daily_with_lags["month"] = daily_with_lags["date"].dt.month
        
        # Fill any remaining NaN values with mean per category
        for cat_id in daily_with_lags["category_id"].unique():
            mask = daily_with_lags["category_id"] == cat_id
            cat_mean = daily_with_lags.loc[mask, "amount"].mean()
            
            for col in ['lag_1', 'lag_7', 'lag_14', 'rolling_7', 'rolling_14']:
                daily_with_lags.loc[mask, col] = daily_with_lags.loc[mask, col].fillna(cat_mean)
        
        daily_with_lags.fillna(0, inplace=True)

        # Feature columns
        FEATURES = [
            "category_id",
            "dow",
            "is_weekend",
            "day",
            "month",
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_7",
            "rolling_14"
        ]

        # -------------------------
        # TRAIN MODEL
        # -------------------------
        X = daily_with_lags[FEATURES]
        y = daily_with_lags["amount"]

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
        # PREDICTION
        # -------------------------
        last_date = daily_with_lags["date"].max()
        results = []
        
        # Get category statistics
        category_stats = {}
        for cat_id in daily_with_lags["category_id"].unique():
            cat_data = daily_with_lags[daily_with_lags["category_id"] == cat_id]
            amounts = cat_data["amount"].values
            
            if len(amounts) > 0:
                stats = {
                    'mean': float(np.mean(amounts)),
                    'std': float(np.std(amounts)) if len(amounts) > 1 else 0,
                    'median': float(np.median(amounts)),
                    'max': float(np.max(amounts)),
                    'min': float(np.min(amounts)),
                    'last_value': float(amounts[-1]) if len(amounts) > 0 else 0,
                    'last_7_avg': float(np.mean(amounts[-7:])) if len(amounts) >= 7 else float(np.mean(amounts)),
                    'last_14_avg': float(np.mean(amounts[-14:])) if len(amounts) >= 14 else float(np.mean(amounts))
                }
            else:
                stats = {
                    'mean': 0,
                    'std': 0,
                    'median': 0,
                    'max': 0,
                    'min': 0,
                    'last_value': 0,
                    'last_7_avg': 0,
                    'last_14_avg': 0
                }
            
            category_stats[cat_id] = stats

        # Generate predictions
        for cat_id in daily_with_lags["category_id"].unique():
            stats = category_stats[cat_id]
            cat_name = le.inverse_transform([cat_id])[0]
            
            # Get recent data for this category
            cat_data = daily_with_lags[daily_with_lags["category_id"] == cat_id]
            recent_amounts = cat_data["amount"].values
            
            # Calculate reasonable bounds
            baseline_mean = stats['mean']
            baseline_std = max(stats['std'], baseline_mean * 0.1)  # Ensure at least 10% of mean
            
            # Set max prediction cap
            max_cap = min(
                baseline_mean + 1.5 * baseline_std,
                baseline_mean * 1.5,
                stats['max'] * 1.2 if stats['max'] > 0 else baseline_mean * 2
            )
            
            # Ensure max_cap is reasonable
            max_cap = max(max_cap, baseline_mean * 1.1)
            
            for i in range(req.days):
                next_date = last_date + timedelta(days=i + 1)
                
                # Use ACTUAL last values for lags (not mean)
                if len(recent_amounts) >= 1:
                    lag_1_val = recent_amounts[-1]
                else:
                    lag_1_val = baseline_mean
                    
                if len(recent_amounts) >= 7:
                    lag_7_val = np.mean(recent_amounts[-7:])
                else:
                    lag_7_val = baseline_mean
                    
                if len(recent_amounts) >= 14:
                    lag_14_val = np.mean(recent_amounts[-14:])
                else:
                    lag_14_val = baseline_mean
                
                # Rolling averages
                rolling_7_val = lag_7_val  # Same as lag_7
                rolling_14_val = lag_14_val  # Same as lag_14
                
                # Prepare features for prediction
                row = {
                    "category_id": cat_id,
                    "dow": next_date.weekday(),
                    "is_weekend": int(next_date.weekday() >= 5),
                    "day": next_date.day,
                    "month": next_date.month,
                    "lag_1": lag_1_val,
                    "lag_7": lag_7_val,
                    "lag_14": lag_14_val,
                    "rolling_7": rolling_7_val,
                    "rolling_14": rolling_14_val,
                }
                
                # Predict
                pred = model.predict(pd.DataFrame([row])[FEATURES])[0]
                
                # Apply safety constraints
                pred = max(0, pred)  # No negative values
                
                # Blend with baseline for stability (70% baseline, 30% prediction)
                blended_pred = 0.7 * baseline_mean + 0.3 * pred
                
                # Apply cap
                blended_pred = min(blended_pred, max_cap)
                
                # Apply decay for future days (slight decay)
                decay_factor = 0.98 ** i
                final_pred = blended_pred * decay_factor
                
                # Ensure minimum prediction
                final_pred = max(final_pred, baseline_mean * 0.3)
                
                results.append({
                    "date": str(next_date.date()),
                    "category": cat_name,
                    "predicted_amount": round(final_pred, 2)
                })

        forecast = pd.DataFrame(results)

        # -------------------------
        # BUILD RESPONSE
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
