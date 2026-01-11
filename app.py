from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import traceback

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Spending Prediction API",
    description="Predict spending for 7, 14, 30, 60, 90 days (category-wise)",
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
    days: int  # Days to predict (7, 14, 30, 60, 90)

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
        
        if req.days not in [7, 14, 30, 60, 90]:
            raise HTTPException(status_code=400, detail="Days must be 7, 14, 30, 60, or 90")

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
        # FILTER: USE ONLY LAST 30 DAYS OF DATA
        # -------------------------
        if len(df) > 0:
            max_date = df["date"].max()
            cutoff_date = max_date - timedelta(days=30)
            df_30days = df[df["date"] >= cutoff_date].copy()
        else:
            df_30days = pd.DataFrame(columns=df.columns)
        
        if df_30days.empty:
            raise HTTPException(status_code=400, detail="No data in last 30 days")

        # -------------------------
        # CALCULATE 30-DAY TOTALS AND PERCENTAGES
        # -------------------------
        # Group by category and sum amounts for last 30 days
        category_totals = df_30days.groupby("category")["amount"].sum()
        
        if category_totals.empty:
            raise HTTPException(status_code=400, detail="No category data in last 30 days")
        
        # Calculate total spending in last 30 days
        total_spent_30days = category_totals.sum()
        
        # Calculate percentage distribution
        category_percentages = (category_totals / total_spent_30days * 100).round(2).to_dict()
        
        # Calculate daily average per category
        category_daily_avg = (category_totals / 30).round(2).to_dict()
        
        # -------------------------
        # CALCULATE PREDICTIONS BASED ON PERCENTAGES
        # -------------------------
        requested_days = req.days
        
        # Method 1: Simple proportional (days/30)
        simple_multiplier = requested_days / 30
        
        # Method 2: Conservative adjustment (accounts for non-linearity)
        # Shorter periods get slightly higher multiplier, longer periods get lower
        if requested_days == 7:
            conservative_multiplier = 0.35  # Instead of 0.233
        elif requested_days == 14:
            conservative_multiplier = 0.55  # Instead of 0.467
        elif requested_days == 30:
            conservative_multiplier = 1.0
        elif requested_days == 60:
            conservative_multiplier = 1.7   # Instead of 2.0
        elif requested_days == 90:
            conservative_multiplier = 2.3   # Instead of 3.0
        
        # Use weighted average of both methods (70% conservative, 30% simple)
        final_multiplier = 0.7 * conservative_multiplier + 0.3 * simple_multiplier
        
        # Calculate predicted total spending
        predicted_total = total_spent_30days * final_multiplier
        
        # Distribute predicted total using category percentages
        predictions_array = []
        
        for category, percentage in category_percentages.items():
            # Calculate amount based on percentage of predicted total
            predicted_amount = predicted_total * (percentage / 100)
            
            # Apply additional conservatism for categories with few transactions
            cat_transactions = df_30days[df_30days["category"] == category]
            
            # If category has very few transactions, reduce prediction further
            transaction_count = len(cat_transactions)
            if transaction_count < 3:
                # Very sparse data - reduce by 50%
                predicted_amount *= 0.5
            elif transaction_count < 7:
                # Limited data - reduce by 25%
                predicted_amount *= 0.75
            
            # Ensure prediction is at least the daily average * requested days
            # but not more than 2x the daily average * requested days
            min_amount = category_daily_avg[category] * requested_days * 0.3  # At least 30% of expected
            max_amount = category_daily_avg[category] * requested_days * 2.0  # At most 2x expected
            
            predicted_amount = max(min_amount, min(predicted_amount, max_amount))
            
            predictions_array.append({
                "category": category,
                "amount": float(round(predicted_amount, 2)),
                "days": requested_days,
                "isDemo": False,
                "percentage": float(percentage),  # For debugging/transparency
                "daily_avg": float(category_daily_avg[category])  # For debugging
            })
        
        # Re-calculate total to ensure consistency
        final_total = sum(item["amount"] for item in predictions_array)
        
        # Optional: If total seems too high/low, scale all predictions proportionally
        # This maintains the percentage distribution while controlling total
        expected_range_min = total_spent_30days * (requested_days / 30) * 0.5  # 50% of linear
        expected_range_max = total_spent_30days * (requested_days / 30) * 1.5  # 150% of linear
        
        if final_total < expected_range_min:
            # Too low - scale up to minimum
            scale_factor = expected_range_min / final_total if final_total > 0 else 1
            for item in predictions_array:
                item["amount"] = float(round(item["amount"] * scale_factor, 2))
        
        elif final_total > expected_range_max:
            # Too high - scale down to maximum
            scale_factor = expected_range_max / final_total
            for item in predictions_array:
                item["amount"] = float(round(item["amount"] * scale_factor, 2))
        
        # Remove debug fields for final response
        for item in predictions_array:
            item.pop("percentage", None)
            item.pop("daily_avg", None)
        
        # Return the exact format frontend expects
        return {
            "predictions": predictions_array,
            "success": True,
            "days": requested_days,
            "total_30days": float(round(total_spent_30days, 2)),  # For reference
            "method": "percentage_distribution"
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
