from flask import Flask,request,jsonify
import re
import joblib
import numpy as np
import requests
import tldextract


app=Flask(__name__)


# Example criteria for fraud detection
def extract_features(email_data):
    """
    Extracts features from email content to match the model's training feature set.
    """
    subject = email_data.get("subject", "")
    body = email_data.get("body", "")
    sender = email_data.get("sender", "")
    links = re.findall(r'http[s]?://\S+', body)
    sender_domain = tldextract.extract(sender).domain
    link_domains = [tldextract.extract(link).domain for link in links]

    # Shortened URL detection
    shortened_services = ["bit.ly", "tinyurl", "goo.gl", "ow.ly", "t.co"]

    # Features aligned with dataset
    features = {
        "having_IP": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', sender) else -1,
        "URL_Length": 1 if len(links[0]) > 75 else -1 if links else 0,
        "Shortining_Service": 1 if any(domain in shortened_services for domain in link_domains) else -1,
        "having_At_Symbol": 1 if "@" in sender else -1,
        "double_slash_redirecting": 1 if "//" in body.split("://")[1] else -1 if links else 0,
        "Prefix_Suffix": 1 if '-' in sender else -1,
        "having_Sub_Domain": 1 if sender_domain.count('.') > 1 else -1,
        "SSLfinal_State": 1 if "https" in sender else -1,
        "Domain_registeration_length": -1,  # Placeholder (Need WHOIS lookup)
        "Favicon": 1 if "<link rel='icon'" in body else -1,
        "port": 1,  # Placeholder (Requires Network Analysis)
        "HTTPS_token": 1 if "https" in sender_domain else -1,
        "Request_URL": 1 if any(domain != sender_domain for domain in link_domains if domain) else -1,
        "URL_of_Anchor": -1 if "click here" in body.lower() else 1,
        "Links_in_tags": 1 if len(links) > 5 else -1,
        "SFH": 1 if "form action=" in body else -1,
        "Submitting_to_email": 1 if "mailto:" in body else -1,
        "Abnormal_URL": 1 if sender_domain == "xyz" else -1,
        "Redirect": 1 if "window.location" in body else -1,
        "on_mouseover": 1 if "onmouseover" in body else -1,
        "RightClick": 1 if "oncontextmenu" in body else -1,
        "popUpWidnow": 1 if "window.open" in body else -1,
        "Iframe": 1 if "<iframe" in body.lower() else -1,
        "age_of_domain": -1,  # Placeholder (Needs WHOIS lookup)
        "DNSRecord": -1,  # Placeholder (Needs WHOIS lookup)
        "web_traffic": -1,  # Placeholder (Requires Alexa API)
        "Page_Rank": -1,  # Placeholder
        "Google_Index": 1 if "google.com" in sender else -1,
        "Links_pointing_to_page": -1 if len(links) < 2 else 1,
    }

    return np.array(list(features.values())).reshape(1, -1)

# Load model and scaler
model=joblib.load('phishing_detector_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/detect_phishing', methods=['POST'])
def detect_phishing():

    email_data=request.json

    # Validate input
    if not email_data:
        return jsonify({"error":"Invalid input. Provide email data."}), 400

    # Extract features & scale them
    features=extract_features(email_data)
    features=scaler.transform(features) # Apply scaling

    # Predict phishing status
    prediction=model.predict(features)[0]
    probability=model.predict_proba(features)[0,1]


    result={
        "is_phishing":bool(prediction),
        "confidence":probability,
        "recommendation": "Avoid clicking links if flagged as phishing."
    }

    return jsonify(result)


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
