from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("lead_scoring_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([{
        "Age": data["age"],
        "Calls": data["calls"],
        "Emails": data["emails"],
        "Source_Ads": 1 if data["source"] == "Ads" else 0,
        "Source_Instagram": 1 if data["source"] == "Instagram" else 0,
        "Source_Referral": 1 if data["source"] == "Referral" else 0,
        "Source_Website": 1 if data["source"] == "Website" else 0
    }])

    score = model.predict_proba(df)[0][1]

    return jsonify({
        "score": round(score * 100,2)
    })

if __name__ == "__main__":
    print("AI server running on port 5001")
    app.run()