import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# TASK-01: DATA DISTRIBUTION VISUALIZATION
# ==========================================

# CSV file
file_path = "student_performance.csv"

# Load dataset
data = pd.read_csv(file_path)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# ==========================================
# DISPLAY COMPLETE DATASET INFORMATION
# ==========================================

print("=" * 60)
print("STUDENT PERFORMANCE DATASET")
print("=" * 60)

print("\nFirst 5 Records:")
print(data.head())

print("\nLast 5 Records:")
print(data.tail())

print("\nTotal Records:", len(data))
print("Total Columns:", len(data.columns))

print("\nColumn Names:")
print(data.columns.tolist())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe(include="all"))


# ==========================================
# IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ==========================================

categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns

numeric_columns = data.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

print("\nNumerical Columns:")
print(numeric_columns.tolist())


# ==========================================
# BAR CHART FOR ALL CATEGORICAL COLUMNS
# ==========================================

for column in categorical_columns:

    print("\nDistribution of", column)
    print(data[column].value_counts())

    plt.figure(figsize=(8, 5))

    data[column].value_counts().plot(
        kind="bar",
        edgecolor="black"
    )

    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Number of Records")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ==========================================
# HISTOGRAM FOR ALL NUMERICAL COLUMNS
# ==========================================

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        data[column].dropna(),
        bins=10,
        edgecolor="black"
    )

    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Number of Records")

    plt.tight_layout()
    plt.show()


# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n" + "=" * 60)
print("TASK-01 COMPLETED SUCCESSFULLY")
print("=" * 60)
 
