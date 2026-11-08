
import pandas as pd

# Load the phishing dataset
df = pd.read_csv("dataset.csv")

# Display basic dataset information
print("PHISHGUARD AI - DATASET INSPECTION")
print("=" * 50)

print("\n1. Dataset dimensions")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n2. Column names")
print(df.columns.tolist())

print("\n3. Dataset information")
df.info()

print("\n4. Missing values")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\n5. Duplicate records")
print("Duplicate rows:", df.duplicated().sum())

print("\n6. Target variable distribution")
print(df["label"].value_counts(dropna=False))

print("\n7. First five records")
print(df.head())

print("\nDataset inspection completed.")
