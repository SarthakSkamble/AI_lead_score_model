from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("lead_scoring_model_final.pkl")
scaler = joblib.load("scaler_final.pkl")
model_columns = joblib.load("model_columns_final.pkl")

KNOWN_SOURCES = [
    "Blog", "Click2Call", "Direct Traffic", "Facebook", "Google", "Live Chat",
    "Nc_Edm", "Olark Chat", "Organic Search", "Pay Per Click Ads", "Press_Release",
    "Reference", "Referral Sites", "Social Media", "Testone", "Welearn",
    "Welearnblog_Home", "Welingak Website", "Youtubechannel"
]
KNOWN_CITIES = [
    "Other Cities", "Other Cities of Maharashtra", "Other Metro Cities",
    "Thane & Outskirts", "Tier II Cities"
]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    
    engagement_score = data.get("engagement_score", 0)
    time_on_site = data.get("time_on_site", 0)
    source = data.get("source", "Unknown")
    city = data.get("city", "Unknown")

    source = str(source).strip().title()
    city = str(city).strip().title() if city else "Unknown"

    
    row = {col: 0 for col in model_columns}

    
    row["Engagement Score"] = engagement_score
    row["Time on Site"] = time_on_site

    
    source_col = f"Lead Source_{source}"
    if source_col in row:
        row[source_col] = 1
    

    city_col = f"City_{city}"
    if city_col in row:
        row[city_col] = 1
   

    
    df = pd.DataFrame([row])[model_columns]

    
    df[["Engagement Score", "Time on Site"]] = scaler.transform(
        df[["Engagement Score", "Time on Site"]]
    )

    
    score = model.predict_proba(df)[0][1]

    return jsonify({
        "score": round(score * 100, 2)
    })


if __name__ == "__main__":
    print("AI server running on port 5001")
    app.run()