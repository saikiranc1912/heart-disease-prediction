import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/heart_disease_model.pkl")

# Example patient data
patient = pd.DataFrame([{
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6
}])

# Make prediction
prediction = model.predict(patient)[0]

print("Prediction:", prediction)

if prediction == 1:
    print("Result: Heart disease detected")
else:
    print("Result: No heart disease detected")