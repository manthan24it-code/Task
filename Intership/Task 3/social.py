import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ==========================================================
# TASK-03: DECISION TREE CLASSIFIER
# SOCIAL NETWORK ADS DATASET
# ==========================================================

print("=" * 75)
print("TASK-03: DECISION TREE CLASSIFIER")
print("SOCIAL NETWORK ADS DATASET")
print("=" * 75)


# ==========================================================
# 1. FIND CSV FILE AUTOMATICALLY
# ==========================================================

folder = os.path.dirname(os.path.abspath(__file__))

csv_files = glob.glob(
    os.path.join(folder, "*.csv")
)

# Look specifically for Social Network Ads CSV
social_files = []

for file in csv_files:

    if "social" in os.path.basename(file).lower():
        social_files.append(file)


if len(social_files) > 0:

    file_path = social_files[0]

elif len(csv_files) > 0:

    file_path = csv_files[0]

else:

    print("\nERROR: No CSV file found.")

    print("\nPlease put your Kaggle CSV file in:")
    print(folder)

    input("\nPress Enter to exit...")
    exit()


print("\nCSV file found:")
print(os.path.basename(file_path))


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

try:

    data = pd.read_csv(
        file_path,
        encoding="utf-8"
    )

except UnicodeDecodeError:

    data = pd.read_csv(
        file_path,
        encoding="latin1"
    )


print("\nDataset loaded successfully!")


# ==========================================================
# 3. CLEAN COLUMN NAMES
# ==========================================================

data.columns = data.columns.str.strip()

print("\n" + "=" * 75)
print("COLUMN NAMES")
print("=" * 75)

print(data.columns.tolist())


# ==========================================================
# 4. DISPLAY ALL RECORDS
# ==========================================================

print("\n" + "=" * 75)
print("ALL DATASET RECORDS")
print("=" * 75)

pd.set_option(
    "display.max_rows",
    None
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print(data.to_string(index=False))


# ==========================================================
# 5. DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print("Total Records:", len(data))
print("Total Columns:", len(data.columns))

print("\nData Types:")

print(data.dtypes)


# ==========================================================
# 6. CHECK MISSING VALUES
# ==========================================================

print("\n" + "=" * 75)
print("MISSING VALUES")
print("=" * 75)

print(data.isnull().sum())

print(
    "\nTotal Missing Values:",
    data.isnull().sum().sum()
)


# ==========================================================
# 7. CHECK DUPLICATES
# ==========================================================

print("\n" + "=" * 75)
print("DUPLICATE RECORDS")
print("=" * 75)

duplicates = data.duplicated().sum()

print(
    "Number of duplicate records:",
    duplicates
)


# Remove duplicates

data = data.drop_duplicates()

print(
    "Records after removing duplicates:",
    len(data)
)


# ==========================================================
# 8. HANDLE MISSING VALUES
# ==========================================================

print("\n" + "=" * 75)
print("DATA CLEANING")
print("=" * 75)


# Numerical columns

numeric_columns = data.select_dtypes(
    include=np.number
).columns


for column in numeric_columns:

    if data[column].isnull().sum() > 0:

        median_value = data[column].median()

        data[column] = data[column].fillna(
            median_value
        )

        print(
            column,
            "filled using median:",
            median_value
        )


# Categorical columns

categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns


for column in categorical_columns:

    if data[column].isnull().sum() > 0:

        mode_value = data[column].mode()[0]

        data[column] = data[column].fillna(
            mode_value
        )

        print(
            column,
            "filled using mode:",
            mode_value
        )


# ==========================================================
# 9. FIND REQUIRED COLUMNS
# ==========================================================

def find_column(name):

    for column in data.columns:

        if column.lower().strip() == name.lower():

            return column

    return None


gender_column = find_column("Gender")

age_column = find_column("Age")

salary_column = find_column("EstimatedSalary")

purchased_column = find_column("Purchased")


print("\n" + "=" * 75)
print("REQUIRED COLUMNS")
print("=" * 75)

print("Gender:", gender_column)
print("Age:", age_column)
print("Estimated Salary:", salary_column)
print("Purchased:", purchased_column)


# ==========================================================
# 10. TARGET DISTRIBUTION
# ==========================================================

if purchased_column is not None:

    print("\n" + "=" * 75)
    print("PURCHASE DISTRIBUTION")
    print("=" * 75)

    print(
        data[purchased_column].value_counts()
    )


# ==========================================================
# 11. ENCODE GENDER
# ==========================================================

if gender_column is not None:

    encoder = LabelEncoder()

    data["Gender_Encoded"] = encoder.fit_transform(
        data[gender_column].astype(str)
    )

    print("\nGender Encoding:")

    for i, value in enumerate(
        encoder.classes_
    ):

        print(
            value,
            "=",
            i
        )


# ==========================================================
# 12. PREPARE FEATURES AND TARGET
# ==========================================================

features = []

if age_column is not None:

    features.append(age_column)


if salary_column is not None:

    features.append(salary_column)


if "Gender_Encoded" in data.columns:

    features.append("Gender_Encoded")


print("\n" + "=" * 75)
print("FEATURES USED FOR MODEL")
print("=" * 75)

print(features)


X = data[features]

y = data[purchased_column]


# ==========================================================
# 13. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 75)
print("TRAIN TEST SPLIT")
print("=" * 75)

print("Total Records:", len(data))

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))


# ==========================================================
# 14. BUILD DECISION TREE
# ==========================================================

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)


# ==========================================================
# 15. TRAIN MODEL
# ==========================================================

model.fit(
    X_train,
    y_train
)


print("\n" + "=" * 75)
print("MODEL TRAINING")
print("=" * 75)

print("Decision Tree model trained successfully!")


# ==========================================================
# 16. PREDICT TEST DATA
# ==========================================================

y_pred = model.predict(
    X_test
)


# ==========================================================
# 17. MODEL ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 75)
print("MODEL ACCURACY")
print("=" * 75)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ==========================================================
# 18. CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================================
# 19. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)


# ==========================================================
# 20. CONFUSION MATRIX HEATMAP
# ==========================================================

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Purchased", "Purchased"],
    yticklabels=["Not Purchased", "Purchased"]
)

plt.title(
    "Confusion Matrix - Decision Tree"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.show()


# ==========================================================
# 21. DECISION TREE VISUALIZATION
# ==========================================================

plt.figure(
    figsize=(20, 10)
)

plot_tree(
    model,
    feature_names=features,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title(
    "Decision Tree Classifier - Social Network Ads"
)

plt.tight_layout()

plt.show()


# ==========================================================
# 22. FEATURE IMPORTANCE
# ==========================================================

importance = model.feature_importances_


feature_importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": importance
    }
)


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 75)
print("FEATURE IMPORTANCE")
print("=" * 75)

print(
    feature_importance.to_string(
        index=False
    )
)


# ==========================================================
# 23. FEATURE IMPORTANCE GRAPH
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"],
    edgecolor="black"
)

plt.title(
    "Feature Importance - Decision Tree"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.show()


# ==========================================================
# 24. ACTUAL VS PREDICTED VALUES
# ==========================================================

comparison = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": y_pred
    }
)


print("\n" + "=" * 75)
print("ACTUAL VS PREDICTED VALUES")
print("=" * 75)

print(
    comparison.to_string(
        index=False
    )
)


# ==========================================================
# 25. PREDICT ALL RECORDS
# ==========================================================

all_predictions = model.predict(
    X
)


data["Predicted_Purchase"] = all_predictions


print("\n" + "=" * 75)
print("PREDICTION FOR ALL RECORDS")
print("=" * 75)

print(
    data.to_string(
        index=False
    )
)


# ==========================================================
# 26. PURCHASE PREDICTION COUNTS
# ==========================================================

print("\n" + "=" * 75)
print("PREDICTED PURCHASE COUNTS")
print("=" * 75)

print(
    pd.Series(
        all_predictions
    ).value_counts()
)


# ==========================================================
# 27. FINAL RESULT
# ==========================================================

print("\n" + "=" * 75)
print("TASK-03 COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "Total Records Processed:",
    len(data)
)

print(
    "Training Records:",
    len(X_train)
)

print(
    "Testing Records:",
    len(X_test)
)

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "\nDecision Tree classification completed."
)

print(
    "Purchase predictions generated for all records."
)
