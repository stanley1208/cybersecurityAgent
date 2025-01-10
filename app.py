from flask import Flask,request,jsonify
import re
from sklearn.ensemble import RandomForestClassifier
import numpy as np


app=Flask(__name__)


# Example criteria for fraud detection
def analyze_account(data):
    """
    Analyzes Instagram account data and returns a fraud probability score.
    """
    # Extract account features
    username=data.get("username","")
    bio=data.get("bio","")
    followers=data.get("followers","")
    following=data.get("following","")
    posts=data.get("posts",0)

    # Feature engineering
    suspicious_keywords=["giveaway","free","click here","winner","dm me"]
    suspicious_bio=any(word in bio.lower() for word in suspicious_keywords)
    suspicious_username=bool(re.search(r"(\d{4,}|free|cash|win)",username.lower()))
    engagement_rate=(posts/max(followers,1))*100    # Avoid division by zero

    # Define simple heuristics for fraud
    is_fraud=(
        suspicious_bio or
        suspicious_username or
        (followers >10000 and posts <5) or
        engagement_rate<1
    )

    return {
        "is_fraud":is_fraud,
        "reason":(
            "Suspicious bio" if suspicious_bio else
            "Suspicious username" if suspicious_username else
            "Low enganemwnt rate" if engagement_rate < 1 else
            "Unusual followers-to-posts ratio"
        )
    }

@app.route('/detect', methods=['POST'])
def detect_fraud():
    # Get account data from request
    account_data=request.json

    # Validate input
    if not account_data:
        return jsonify({"error":"Invalid input. Provide account data."}), 400

    # Analyze account
    result=analyze_account(account_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)