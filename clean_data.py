import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "data/heart_disease.csv"

columns = [
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
    "thal",
    "target"
]

data = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?"
)

print("Missing values after conversion:")
print(data.isnull().sum())

data = data.dropna()

print("\nDataset shape after removing missing values:")
print(data.shape)

data["target"] = (data["target"] > 0).astype(int)

print("\nTarget values after conversion:")
print(data["target"].value_counts())

X = data.drop("target", axis=1)
y = data["target"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features shape:")
print(X_train.shape)

print("\nTesting features shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)