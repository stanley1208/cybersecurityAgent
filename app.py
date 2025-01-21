from flask import Flask,request,jsonify
import re
import joblib
import numpy as np
import requests
import tldextract


app=Flask(__name__)

# Load a pre-trained phishing detection model (placeholder)
model=joblib.load('phishing_detector_model.pkl')    # Replace with actual model

# Example criteria for fraud detection
def extract_features(email_data):
    """
    Extract features from email content to match the model's feature set.
    """
    subject = email_data.get("subject", "")
    body = email_data.get("body", "")
    sender = email_data.get("sender", "")
    links = re.findall(r'http[s]?://\S+', body)
    sender_domain = tldextract.extract(sender).domain
    link_domains = [tldextract.extract(link).domain for link in links]

    # Updated feature set (placeholder for all 30 features)
    features = {
        "has_suspicious_links": any("@" not in link.split("/")[2] for link in links),
        "urgent_language": any(word in body.lower() for word in ["urgent", "act now", "verify"]),
        "unusual_sender": sender.endswith("xyz") or sender.startswith("no-reply"),
        "link_domain_mismatch": any(domain != sender_domain for domain in link_domains if domain),
        "empty_subject": len(subject.strip()) == 0,
        # Add placeholders or compute the rest of the required features
        **{f"feature_{i}": 0 for i in range(5, 30)},  # Replace with actual logic
    }
    return np.array(list(features.values())).reshape(1, -1)



@app.route('/detect_phishing', methods=['POST'])
def detect_phishing():

    email_data=request.json

    # Validate input
    if not email_data:
        return jsonify({"error":"Invalid input. Provide email data."}), 400

    required_field=["subject","body","sender"]
    missing_field=[field for field in required_field if field not in email_data]

    if missing_field:
        return jsonify({"error":f"Missing Field: {','.join(missing_field)}"}), 400


    # Extract feature
    features=extract_features(email_data)

    # Predict phishing status
    prediction=model.predict(features)[0]
    probability=model.predict_proba(features)[0,1]

    # Generate explanation
    reason=[]
    if features[0,0]:   # has_suspicious_links
        reason.append("Suspicious links detected.")
    if features[0,1]:   # urgent_language
        reason.append("Urgent language used.")
    if features[0,2]:   # unusual_sender
        reason.append("Sender domain appears suspicious.")

    result={
        "is_phishing":bool(prediction),
        "confidence":probability,
        "reason":reason if reason else "No significant phishing indicators detected.",
        "recommendation": "Do not click any links or share sensitive information if flagged as phishing."

    }

    return jsonify(result)


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
