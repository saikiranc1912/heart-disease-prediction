from flask import Flask, request, jsonify
import joblib
import pandas as pd

REQUIRED_FIELDS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]
NUMERIC_FIELDS = REQUIRED_FIELDS
app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("model/heart_disease_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body is required"
        }), 400

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "fields": missing_fields
        }), 400

    for field in NUMERIC_FIELDS:
        if not isinstance(data[field], (int, float)):
            return jsonify({
                "error": "Invalid field value",
                "field": field,
                "message": "Value must be numeric"
            }), 400

    patient = pd.DataFrame([data])

    prediction = model.predict(patient)[0]

    if prediction == 1:
        result = "Heart disease detected"
    else:
        result = "No heart disease detected"

    return jsonify({
        "prediction": int(prediction),
        "result": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)