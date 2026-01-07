from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

app = FastAPI(
    title="Spending Prediction API",
    description="Predict spending for 7, 14, 30 days (category-wise)",
    version="1.0"
)

# =========================
# REQUEST MODELS
# =========================
class Transaction(BaseModel):
    date: str          # YYYY-MM-DD
    amount: float
    category: str

class PredictRequest(BaseModel):
    transactions: List[Transaction]

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

    if not req.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    df = pd.DataFrame([t.dict() for t in req.transactions])

    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format")

    df["amount"] = df["amount"].abs()

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data")

    # Daily totals
    daily = (
        df.groupby(["date", "category"], as_index=False)["amount"]
        .sum()
        .sort_values("date")
    )

    if daily["date"].nunique() < 5:
        raise HTTPException(
            status_code=400,
            detail="Not enough data (minimum 5 days required)"
        )

    # Encode categories
    le = LabelEncoder()
    daily["category_id"] = le.fit_transform(daily["category"])

    daily = daily.sort_values(["category_id", "date"])

    # Date features
    daily["dow"] = daily["date"].dt.weekday
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["day"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month

    # =========================
    # LAG FEATURES (FIXED)
    # =========================
    daily["lag_1"] = daily.groupby("category_id")["amount"].shift(1)

    daily["lag_7"] = daily.groupby("category_id")["amount"].transform(
        lambda x: x.rolling(7, min_periods=1).mean().shift(1)
    )

    daily["lag_14"] = daily.groupby("category_id")["amount"].transform(
        lambda x: x.rolling(14, min_periods=1).mean().shift(1)
    )

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

    X = daily[FEATURES]
    y = daily["amount"]

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)

    # =========================
    # FUTURE PREDICTION
    # =========================
    last_date = daily["date"].max()
    results = []

    for cat_id in daily["category_id"].unique():
        history = daily[daily["category_id"] == cat_id].copy()

        for i in range(30):
            next_date = last_date + timedelta(days=i + 1)

            last_amt = history["amount"].iloc[-1] if not history.empty else 0

            row = {
                "category_id": cat_id,
                "dow": next_date.weekday(),
                "is_weekend": int(next_date.weekday() >= 5),
                "day": next_date.day,
                "month": next_date.month,
                "lag_1": last_amt,
                "lag_7": history.tail(7)["amount"].mean() if not history.empty else 0,
                "lag_14": history.tail(14)["amount"].mean() if not history.empty else 0,
            }

            pred = max(model.predict(pd.DataFrame([row])[FEATURES])[0], 0)

            history = pd.concat(
                [history, pd.DataFrame([{
                    "date": next_date,
                    "category_id": cat_id,
                    "amount": pred
                }])],
                ignore_index=True
            )

            results.append({
                "date": str(next_date.date()),
                "category": le.inverse_transform([cat_id])[0],
                "predicted_amount": round(pred, 2)
            })

    forecast = pd.DataFrame(results)

    output: Dict[str, Dict[str, float]] = {}
    for days in [7, 14, 30]:
        temp = forecast.groupby("category").head(days)
        output[f"{days}_days"] = (
            temp.groupby("category")["predicted_amount"]
            .sum()
            .round(2)
            .to_dict()
        )

    return {"predictions": output}
