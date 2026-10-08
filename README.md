# Heart Disease Prediction

A simple applied machine learning project that predicts the presence or absence of heart disease using a clinical dataset, Scikit-learn, Logistic Regression, and a Flask REST API.

This project was developed as an academic and portfolio project to demonstrate the practical application of machine learning concepts including data preprocessing, supervised classification, model evaluation, model persistence, and REST API development.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not intended to provide medical advice, diagnosis, or clinical decision-making.

---

## Project Overview

The goal of this project is to build a simple machine learning classification system that predicts whether heart disease is present based on patient-related clinical attributes.

The project uses the UCI Heart Disease dataset and follows a straightforward machine learning workflow:

```text
Clinical Dataset
       |
       v
Data Inspection
       |
       v
Data Cleaning
       |
       v
Target Conversion
       |
       v
Train/Test Split
       |
       v
Logistic Regression
       |
       v
Model Evaluation
       |
       v
Saved ML Model
       |
       v
Flask REST API
       |
       v
Prediction
```

---

## Technologies Used

- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- StandardScaler
- Joblib
- Flask
- Git / GitHub
- Postman

---

## Dataset

The project uses the **UCI Heart Disease dataset**, specifically the processed Cleveland dataset.

The original dataset contains:

- 303 records
- 14 columns
- 13 input features
- 1 target variable

### Features

The 13 input features used by the model are:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

### Target

The original dataset contains target values from `0` to `4`.

For this project, the target was converted into a binary classification:

```text
0     -> No heart disease
1-4   -> Heart disease
```

After removing incomplete records containing missing values:

```text
Records: 297
Features: 13
```

Target distribution:

```text
No heart disease: 160
Heart disease:    137
```

---

## Data Preprocessing

The dataset required basic preprocessing before model training.

### Missing Values

The original dataset contains `?` values in some fields.

These values were converted to missing values using Pandas, and incomplete records were removed.

Six records containing missing values were removed.

```python
data = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?"
)

data = data.dropna()
```

### Target Conversion

The original target values were converted into binary values:

```python
data["target"] = (data["target"] > 0).astype(int)
```

### Train/Test Split

The cleaned dataset was split into:

```text
80% Training Data
20% Testing Data
```

The split used stratification to preserve the class distribution between the training and testing datasets.

---

## Machine Learning Model

The project uses **Logistic Regression** for binary classification.

A Scikit-learn `Pipeline` was used to combine feature scaling and classification:

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

The `StandardScaler` standardizes the input features before they are passed to the Logistic Regression classifier.

The model was trained using the training dataset and evaluated using the test dataset.

The trained model was saved using Joblib:

```text
model/heart_disease_model.pkl
```

---

## Model Evaluation

The trained model achieved the following result on the test dataset:

### Test Accuracy

```text
83.33%
```

The test dataset contained 60 records, of which 50 were classified correctly.

### Confusion Matrix

```text
[[28  4]
 [ 6 22]]
```

This represents:

```text
True Negatives: 28
False Positives: 4
False Negatives: 6
True Positives: 22
```

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.82      0.88      0.85        32
           1       0.85      0.79      0.81        28

    accuracy                           0.83        60
   macro avg       0.83      0.83      0.83        60
weighted avg       0.83      0.83      0.83        60
```

The evaluation demonstrates that the trained model can perform binary classification on the test dataset.

---

## Prediction Script

The project includes `predict.py`, which loads the saved machine learning model and performs a prediction using sample patient data.

### Example Input

```python
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
```

The saved model produced:

```text
Prediction: 0
Result: No heart disease detected
```

---

## Flask Prediction API

The trained machine learning model is exposed through a simple Flask REST API.

### Endpoint

```text
POST /predict
```

### Local URL

```text
http://localhost:5000/predict
```

### Request

Example JSON request:

```json
{
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
}
```

### Successful Response

```json
{
    "prediction": 0,
    "result": "No heart disease detected"
}
```

---

## API Validation

The Flask API includes basic request validation to handle invalid requests.

### Missing Required Field

If a required field is missing, the API returns:

```json
{
    "error": "Missing required fields",
    "fields": [
        "chol"
    ]
}
```

HTTP status:

```text
400 Bad Request
```

### Invalid Data Type

If a field contains a non-numeric value, the API returns:

```json
{
    "error": "Invalid field value",
    "field": "age",
    "message": "Value must be numeric"
}
```

HTTP status:

```text
400 Bad Request
```

These validation scenarios were tested using Postman.

---

## Project Structure

```text
heart-disease-prediction/
│
├── data/
│   ├── heart-disease.names
│   └── heart_disease.csv
│
├── model/
│   └── heart_disease_model.pkl
│
├── prediction-service/
│   └── app.py
│
├── clean_data.py
├── inspect_data.py
├── inspect_dataset.py
├── predict.py
├── train_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/saikiranc1912/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

On Git Bash:

```bash
source venv/Scripts/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Prediction Script

```bash
python predict.py
```

### 6. Start the Flask API

```bash
python prediction-service/app.py
```

The API will run on:

```text
http://localhost:5000
```

### 7. Test the API

Use Postman or another REST client.

Send a `POST` request to:

```text
http://localhost:5000/predict
```

with the required JSON fields described in the Flask Prediction API section.

---

## Model Training

To retrain the machine learning model:

```bash
python train_model.py
```

The script:

1. Loads the dataset
2. Handles missing values
3. Converts the target into binary classification
4. Splits the data into training and testing sets
5. Scales the features
6. Trains Logistic Regression
7. Evaluates the model
8. Saves the trained model

The trained model is saved to:

```text
model/heart_disease_model.pkl
```

---

## Key Learning Outcomes

This project demonstrates practical experience with:

- Data loading and inspection
- Data cleaning and preprocessing
- Handling missing values
- Binary classification
- Logistic Regression
- Feature scaling
- Train/test splitting
- Model evaluation
- Confusion matrix analysis
- Classification metrics
- Model persistence using Joblib
- Building a REST API with Flask
- Request validation
- REST API testing using Postman

---

## Future Improvements

Possible future enhancements include:

- Comparing additional classification algorithms
- Improving validation and error handling
- Adding automated unit tests
- Creating a web-based user interface
- Containerizing the application
- Deploying the prediction API to the cloud

---

## Disclaimer

This project is intended for educational and portfolio purposes only.

The predictions generated by this application should **not** be considered medical advice, diagnosis, or a substitute for professional medical evaluation.
