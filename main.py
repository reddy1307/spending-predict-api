from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

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

    # -------------------------
    # Convert to DataFrame
    # -------------------------
    df = pd.DataFrame([t.dict() for t in req.transactions])
    df["date"] = pd.to_datetime(df["date"])

    # Only expenses
    df = df[df["amount"] > 0]

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid expense data")

    # -------------------------
    # DAILY AGGREGATION
    # -------------------------
    daily = (
        df.groupby(["date", "category"])["amount"]
        .sum()
        .reset_index()
    )

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
    # FEATURE ENGINEERING
    # -------------------------
    daily = daily.sort_values(["category_id", "date"])

    daily["dow"] = daily["date"].dt.weekday
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["day"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month

    daily["lag_1"] = daily.groupby("category_id")["amount"].shift(1)
    daily["lag_7"] = (
        daily.groupby("category_id")["amount"]
        .rolling(7)
        .mean()
        .shift(1)
    )
    daily["lag_14"] = (
        daily.groupby("category_id")["amount"]
        .rolling(14)
        .mean()
        .shift(1)
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

    # -------------------------
    # TRAIN MODEL
    # -------------------------
    X = daily[FEATURES]
    y = daily["amount"]

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)

    # -------------------------
    # ROLLING PREDICTION (30 DAYS)
    # -------------------------
    last_date = daily["date"].max()
    results = []

    for cat_id in daily["category_id"].unique():
        history = daily[daily["category_id"] == cat_id].copy()

        for i in range(30):
            next_date = last_date + timedelta(days=i + 1)

            row = {
                "category_id": cat_id,
                "dow": next_date.weekday(),
                "is_weekend": int(next_date.weekday() >= 5),
                "day": next_date.day,
                "month": next_date.month,
                "lag_1": history.iloc[-1]["amount"],
                "lag_7": history.tail(7)["amount"].mean(),
                "lag_14": history.tail(14)["amount"].mean(),
            }

            pred = max(
                model.predict(pd.DataFrame([row])[FEATURES])[0],
                0
            )

            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [{
                            "date": next_date,
                            "category_id": cat_id,
                            "amount": pred
                        }]
                    )
                ],
                ignore_index=True
            )

            results.append({
                "date": str(next_date.date()),
                "category": le.inverse_transform([cat_id])[0],
                "predicted_amount": round(pred, 2)
            })

    forecast = pd.DataFrame(results)

    # -------------------------
    # AGGREGATE 7 / 14 / 30
    # -------------------------
    output: Dict[str, Dict[str, float]] = {}

    for days in [7, 14, 30]:
        output[f"{days}_days"] = (
            forecast.groupby("category")["predicted_amount"]
            .head(days)
            .groupby(level=0)
            .sum()
            .round(2)
            .to_dict()
        )

    return {
        "predictions": output
    }
