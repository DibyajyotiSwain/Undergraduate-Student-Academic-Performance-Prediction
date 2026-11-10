import pandas as pd
from pathlib import Path

# Dataset path
DATA_PATH = Path("data/student_performance_synthetic.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 55)
print("UNDERGRADUATE STUDENT PERFORMANCE DATASET")
print("=" * 55)

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. FIRST FIVE RECORDS")
print(df.head())

print("\n3. COLUMN INFORMATION")
print(df.info())

print("\n4. MISSING VALUES")
print(df.isnull().sum())

print("\n5. DUPLICATE RECORDS")
print(df.duplicated().sum())

print("\n6. STATISTICAL SUMMARY")
print(df.describe())

print("\n7. STUDENTS AND SEMESTERS")
print("Unique students:", df["student_id"].nunique())
print("Unique semesters:", df["semester"].nunique())

print("\n8. ACADEMIC YEAR DISTRIBUTION")
print(df["academic_year"].value_counts().sort_index())

print("\n" + "=" * 55)
print("DATASET VALIDATION COMPLETED")
print("=" * 55)