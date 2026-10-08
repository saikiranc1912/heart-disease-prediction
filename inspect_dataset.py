import pandas as pd

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

data = pd.read_csv(file_path, header=None, names=columns)

print("Dataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nQuestion mark values:")
print(data.isin(["?"]).sum())

print("\nTarget values:")
print(data["target"].value_counts())