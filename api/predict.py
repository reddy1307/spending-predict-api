import json
from model.predictor import predict_spending

def handler(request):
    # Allow only POST
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Method not allowed"})
        }

    try:
        body = json.loads(request.body)

        transactions = body.get("transactions", [])
        if not transactions:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No transactions provided"})
            }

        predictions = predict_spending(transactions)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "predictions": predictions
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
