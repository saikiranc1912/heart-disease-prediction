import pandas as pd

file_path = "data/heart_disease.csv"

data = pd.read_csv(file_path, header=None)

print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

print("\nFirst 5 rows:")
print(data.head())