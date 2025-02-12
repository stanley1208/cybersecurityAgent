from flask import Flask, request, jsonify, render_template
import re
import joblib
import numpy as np
import tldextract
from urllib.parse import urlparse
import requests
from datetime import datetime
import whois
import html

app = Flask(__name__)

# Load model and scaler
model = joblib.load('phishing_detector_model.pkl')
scaler = joblib.load('scaler.pkl')


def is_ip_address(url):
    """Check if the URL contains an IP address."""
    ip_pattern = re.compile(
        r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    )
    return bool(ip_pattern.search(url))


def check_domain_age(domain):
    """Check domain age in days."""
    try:
        w = whois.whois(domain)
        if w.creation_date:
            creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            age = (datetime.now() - creation_date).days
            return age
    except:
        return -1
    return -1


def check_ssl_certificate(url):
    """Verify SSL certificate."""
    try:
        response = requests.get(url, verify=True, timeout=5)
        return 1 if response.ok else -1
    except:
        return -1


def extract_features(email_data):
    """Enhanced feature extraction with better error handling and validation."""
    subject = html.escape(email_data.get("subject", ""))
    body = html.escape(email_data.get("body", ""))
    sender = html.escape(email_data.get("sender", ""))

    # Extract and validate URLs
    # Extract URLs from email body safely
    links = []
    for match in re.findall(r'http[s]?://\S+', body):
        try:
            parsed_url = urlparse(match)
            if parsed_url.netloc:  # Ensure it's a valid URL
                links.append(parsed_url.geturl())
        except ValueError:
            continue  # Skip invalid URLs

    links = [link for link in links if urlparse(link).netloc]  # Valid URLs only

    sender_domain = tldextract.extract(sender).domain
    link_domains = [tldextract.extract(link).domain for link in links]

    # Enhanced shortened URL detection
    shortened_services = {
        "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "tiny.cc",
        "is.gd", "cli.gs", "pic.gd", "DwarfURL.com", "yfrog.com", "migre.me"
    }

    features = {
        "having_IP": 1 if any(is_ip_address(url) for url in links + [sender]) else -1,
        "URL_Length": 1 if any(len(link) > 75 for link in links) else -1 if links else 0,
        "Shortining_Service": 1 if any(domain in shortened_services for domain in link_domains) else -1,
        "having_At_Symbol": 1 if "@" in sender or any("@" in link for link in links) else -1,
        "double_slash_redirecting": 1 if any("//" in link.split("://")[1] for link in links if "://" in link) else -1,
        "Prefix_Suffix": 1 if any("-" in domain for domain in [sender_domain] + link_domains) else -1,
        "having_Sub_Domain": 1 if any(domain.count('.') > 1 for domain in [sender_domain] + link_domains) else -1,
        "SSLfinal_State": check_ssl_certificate(links[0]) if links else -1,
        "Domain_registeration_length": 1 if check_domain_age(sender_domain) < 365 else -1,
        "Favicon": 1 if any(('<link rel="icon"' in body, '<link rel="shortcut icon"' in body)) else -1,
        "port": 1 if any(":80/" in link or ":443/" in link for link in links) else -1,
        "HTTPS_token": 1 if "https" in sender_domain or any("https" in domain for domain in link_domains) else -1,
        "Request_URL": 1 if len(set(link_domains) - {sender_domain}) > 0 else -1,
        "URL_of_Anchor": -1 if re.search(r'<a\s+[^>]*>click here|here|this|more</a>', body, re.I) else 1,
        "Links_in_tags": 1 if len(links) > 5 else -1,
        "SFH": 1 if '<form' in body and 'action=' in body else -1,
        "Submitting_to_email": 1 if 'mailto:' in body or 'mail()' in body else -1,
        "Abnormal_URL": 1 if sender_domain not in link_domains and links else -1,
        "Redirect": 1 if any(('window.location' in body, 'document.location' in body, 'document.href' in body)) else -1,
        "on_mouseover": 1 if 'onmouseover=' in body else -1,
        "RightClick": 1 if 'oncontextmenu=' in body or 'event.button==2' in body else -1,
        "popUpWidnow": 1 if 'window.open' in body else -1,
        "Iframe": 1 if '<iframe' in body or '<frame' in body else -1,
        "age_of_domain": 1 if check_domain_age(sender_domain) > 180 else -1,
        "DNSRecord": 1 if sender_domain else -1,  # Simplified DNS check
        "web_traffic": -1,  # Would require external API
        "Page_Rank": -1,  # Would require external API
        "Google_Index": 1 if sender_domain else -1,  # Simplified check
        "Links_pointing_to_page": 1 if len(links) > 2 else -1,


    }
    print("Extracted Features:", features)
    print("Total Features Extracted:", len(features))
    return np.array(list(features.values())).reshape(1, -1)


@app.route('/')
def home():
    return render_template('index.html')


# In your app.py, modify the detect_phishing route:
@app.route('/detect_phishing/', methods=['POST'])
def detect_phishing():
    try:
        email_data = {
            "subject": request.form.get("subject", "").strip(),
            "body": request.form.get("body", "").strip(),
            "sender": request.form.get("sender", "").strip(),
        }

        features = extract_features(email_data)
        features_scaled = scaler.transform(features)

        # Get prediction probability and convert to percentage
        probability = model.predict_proba(features_scaled)[0, 1]
        confidence_percentage = round(probability * 100, 2)  # Convert to percentage with 2 decimal places

        threshold = 0.4
        is_phishing = probability > threshold

        result = {
            "is_phishing": is_phishing,
            "confidence": confidence_percentage,  # Now sending as percentage
            "recommendation": "⚠️ Avoid clicking links! This email seems suspicious." if is_phishing else "✅ Email looks safe.",
        }

        return render_template('result.html', result=result)

    except Exception as e:
        print(f"Error: {str(e)}")  # Add this for debugging
        return render_template('result.html',
                               result={"error": f"An error occurred: {str(e)}",
                                       "is_phishing": True,
                                       "confidence": 0,
                                       "recommendation": "Error during analysis - treat with caution."})
if __name__ == '__main__':
    app.run(debug=True)