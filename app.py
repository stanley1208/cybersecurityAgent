from flask import Flask,request,jsonify
import re
import joblib
import numpy as np


app=Flask(__name__)

# Load a pre-trained phishing detection model (placeholder)
model=joblib.load('phishing_detector_model.pkl')    # Replace with actual model

# Example criteria for fraud detection
def extract_features(email_data):
    """
    Extracts features from email content for phishing detection.
    """
    # Extract account features
    subject=email_data.get("subject","")
    body=email_data.get("body","")
    sender=email_data.get("sender","")
    links=re.findall(r'http[s]?://\S+',body)

    # Example features
    features={
        "has_suspicious_links":any("@" not in link.split("/")[2] for link in links),
        "urgent_language":any(word in body.lower() for word in ["urgent","act now","verify"]),
        "unusual_sender":sender.endswith("xyz") or sender.startswith("no-reply"),
    }
    return np.array(list(features.values())).reshape(1,-1)






@app.route('/detect_phishing', methods=['POST'])
def detect_phishing():

    email_data=request.json

    # Validate input
    if not email_data:
        return jsonify({"error":"Invalid input. Provide email data."}), 400

    features=extract_features(email_data)
    prediction=model.predict(features)[0]
    probability=model.predict_proba(features)[0,1]

    result={
        "is_phishing":bool(prediction),
        "confidence":probability,
        "reason":"Detected phishing indicators in the email." if prediction else "No significant phishing indicators detected.",
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)