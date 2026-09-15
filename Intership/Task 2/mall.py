import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# TASK-02: MALL CUSTOMER DATA ANALYSIS
# ==========================================================

print("=" * 70)
print("TASK-02: MALL CUSTOMER DATA ANALYSIS")
print("=" * 70)

# ==========================================================
# 1. FIND CSV AUTOMATICALLY
# ==========================================================

folder = os.path.dirname(os.path.abspath(__file__))

csv_files = glob.glob(os.path.join(folder, "*.csv"))

if len(csv_files) == 0:
    print("\nERROR: No CSV file found!")
    print("\nPlease put your Kaggle CSV file in this folder:")
    print(folder)
    input("\nPress Enter to exit...")
    exit()

# Use the first CSV file found
file_path = csv_files[0]

print("\nCSV file found:")
print(os.path.basename(file_path))

# ==========================================================
# 2. LOAD DATASET
# ==========================================================

try:
    data = pd.read_csv(file_path, encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding="latin1")

print("\nDataset loaded successfully!")

# Clean column names
data.columns = data.columns.str.strip()

# ==========================================================
# 3. BASIC INFORMATION
# ==========================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nTotal Records:", len(data))
print("Total Columns:", len(data.columns))

print("\nColumn Names:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

# ==========================================================
# 4. DISPLAY ALL RECORDS
# ==========================================================

print("\n" + "=" * 70)
print("ALL RECORDS")
print("=" * 70)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(data.to_string(index=False))

# ==========================================================
# 5. FIRST AND LAST RECORDS
# ==========================================================

print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

print(data.head())

print("\n" + "=" * 70)
print("LAST 5 RECORDS")
print("=" * 70)

print(data.tail())

# ==========================================================
# 6. MISSING VALUES BEFORE CLEANING
# ==========================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 70)

print(data.isnull().sum())

print("\nTotal Missing Values:",
      data.isnull().sum().sum())

# ==========================================================
# 7. DUPLICATE RECORDS
# ==========================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicates = data.duplicated().sum()

print("Duplicate Records:", duplicates)

# ==========================================================
# 8. REMOVE DUPLICATES
# ==========================================================

data = data.drop_duplicates()

print("\nAfter removing duplicates:")
print("Total Records:", len(data))

# ==========================================================
# 9. HANDLE MISSING VALUES
# ==========================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

numeric_columns = data.select_dtypes(
    include=np.number
).columns

categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns

# Numerical missing values
for column in numeric_columns:

    if data[column].isnull().sum() > 0:

        median_value = data[column].median()

        data[column] = data[column].fillna(
            median_value
        )

        print(
            column,
            "missing values filled using median:",
            median_value
        )

# Categorical missing values
for column in categorical_columns:

    if data[column].isnull().sum() > 0:

        mode_value = data[column].mode()[0]

        data[column] = data[column].fillna(
            mode_value
        )

        print(
            column,
            "missing values filled using mode:",
            mode_value
        )

# ==========================================================
# 10. MISSING VALUES AFTER CLEANING
# ==========================================================

print("\n" + "=" * 70)
print("MISSING VALUES AFTER CLEANING")
print("=" * 70)

print(data.isnull().sum())

print("\nTotal Missing Values:",
      data.isnull().sum().sum())

# ==========================================================
# 11. STATISTICAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(data.describe(include="all"))

# ==========================================================
# 12. CATEGORICAL DATA ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("CATEGORICAL DATA ANALYSIS")
print("=" * 70)

for column in categorical_columns:

    print("\n", column)
    print(data[column].value_counts())

# ==========================================================
# 13. NUMERICAL DATA ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("NUMERICAL DATA ANALYSIS")
print("=" * 70)

print(data[numeric_columns].describe())

# ==========================================================
# 14. GENDER DISTRIBUTION
# ==========================================================

gender_column = None

for column in data.columns:

    if column.lower().strip() == "gender":
        gender_column = column
        break

if gender_column:

    print("\n" + "=" * 70)
    print("GENDER DISTRIBUTION")
    print("=" * 70)

    print(data[gender_column].value_counts())

    plt.figure(figsize=(8, 5))

    data[gender_column].value_counts().plot(
        kind="bar",
        edgecolor="black"
    )

    plt.title("Customer Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")

    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()

# ==========================================================
# 15. FIND AGE COLUMN
# ==========================================================

age_column = None

for column in data.columns:

    if column.lower().strip() == "age":
        age_column = column
        break

# ==========================================================
# 16. AGE DISTRIBUTION
# ==========================================================

if age_column:

    plt.figure(figsize=(8, 5))

    plt.hist(
        data[age_column].dropna(),
        bins=10,
        edgecolor="black"
    )

    plt.title("Distribution of Customer Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()

# ==========================================================
# 17. FIND INCOME COLUMN
# ==========================================================

income_column = None

for column in data.columns:

    if "income" in column.lower():

        income_column = column
        break

# ==========================================================
# 18. ANNUAL INCOME DISTRIBUTION
# ==========================================================

if income_column:

    plt.figure(figsize=(8, 5))

    plt.hist(
        data[income_column].dropna(),
        bins=10,
        edgecolor="black"
    )

    plt.title("Distribution of Annual Income")
    plt.xlabel(income_column)
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()

# ==========================================================
# 19. FIND SPENDING SCORE COLUMN
# ==========================================================

spending_column = None

for column in data.columns:

    if "spending" in column.lower():

        spending_column = column
        break

# ==========================================================
# 20. SPENDING SCORE DISTRIBUTION
# ==========================================================

if spending_column:

    plt.figure(figsize=(8, 5))

    plt.hist(
        data[spending_column].dropna(),
        bins=10,
        edgecolor="black"
    )

    plt.title("Distribution of Spending Score")
    plt.xlabel(spending_column)
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()

# ==========================================================
# 21. INCOME VS SPENDING SCORE
# ==========================================================

if income_column and spending_column:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data[income_column],
        data[spending_column],
        alpha=0.7
    )

    plt.title("Annual Income vs Spending Score")
    plt.xlabel(income_column)
    plt.ylabel(spending_column)

    plt.tight_layout()
    plt.show()

# ==========================================================
# 22. AGE VS SPENDING SCORE
# ==========================================================

if age_column and spending_column:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data[age_column],
        data[spending_column],
        alpha=0.7
    )

    plt.title("Age vs Spending Score")
    plt.xlabel("Age")
    plt.ylabel("Spending Score")

    plt.tight_layout()
    plt.show()

# ==========================================================
# 23. AGE VS INCOME
# ==========================================================

if age_column and income_column:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data[age_column],
        data[income_column],
        alpha=0.7
    )

    plt.title("Age vs Annual Income")
    plt.xlabel("Age")
    plt.ylabel("Annual Income")

    plt.tight_layout()
    plt.show()

# ==========================================================
# 24. CORRELATION MATRIX
# ==========================================================

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

correlation_columns = []

if age_column:
    correlation_columns.append(age_column)

if income_column:
    correlation_columns.append(income_column)

if spending_column:
    correlation_columns.append(spending_column)

if len(correlation_columns) >= 2:

    correlation = data[
        correlation_columns
    ].corr()

    print(correlation)

# ==========================================================
# 25. CORRELATION HEATMAP
# ==========================================================

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()

# ==========================================================
# 26. AVERAGE VALUES BY GENDER
# ==========================================================

if gender_column:

    group_columns = []

    if age_column:
        group_columns.append(age_column)

    if income_column:
        group_columns.append(income_column)

    if spending_column:
        group_columns.append(spending_column)

    if group_columns:

        print("\n" + "=" * 70)
        print("AVERAGE VALUES BY GENDER")
        print("=" * 70)

        result = data.groupby(
            gender_column
        )[group_columns].mean()

        print(result)

# ==========================================================
# 27. FINAL CLEANED DATASET
# ==========================================================

print("\n" + "=" * 70)
print("FINAL CLEANED DATASET")
print("=" * 70)

print(data.to_string(index=False))

# ==========================================================
# 28. FINAL RESULT
# ==========================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("Total Records Processed:", len(data))
print("Total Columns:", len(data.columns))

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

print("\n" + "=" * 70)
print("TASK-02 COMPLETED SUCCESSFULLY")
print("=" * 70)
