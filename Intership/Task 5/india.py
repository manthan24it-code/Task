# ============================================================
# TASK-05
# INDIAN CAR ACCIDENT ANALYSIS
# Dataset: India Car Accident Analysis Dataset (2022-2023)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# ============================================================
# 1. FIND CSV FILE AUTOMATICALLY
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))

csv_files = glob.glob(
    os.path.join(folder, "*.csv")
)

if len(csv_files) == 0:

    print("ERROR: No CSV file found.")
    print()
    print("Please put your Kaggle CSV file")
    print("in the same folder as media.py")

    input("\nPress Enter to close...")
    exit()

print("=" * 70)
print("             TASK-05: TRAFFIC ACCIDENT ANALYSIS")
print("=" * 70)

print("\nCSV file found:")
print(csv_files[0])

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(csv_files[0])

print("\nDataset loaded successfully!")

print("Total records:", len(df))
print("Total columns:", len(df.columns))

# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

print(df.columns.tolist())

# ============================================================
# 4. FIRST 10 RECORDS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 RECORDS")
print("=" * 70)

print(df.head(10).to_string())

# ============================================================
# 5. LAST 10 RECORDS
# ============================================================

print("\n" + "=" * 70)
print("LAST 10 RECORDS")
print("=" * 70)

print(df.tail(10).to_string())

# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(df.info())

# ============================================================
# 7. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe(include="all").to_string())

# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing.to_string())

# ============================================================
# 9. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate records:", duplicate_count)

# ============================================================
# 10. REMOVE DUPLICATES
# ============================================================

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        "Records after removing duplicates:",
        len(df)
    )

else:

    print("No duplicate records found.")

# ============================================================
# 11. DATE PROCESSING
# ============================================================

if "inverse_data" in df.columns:

    df["inverse_data"] = pd.to_datetime(
        df["inverse_data"],
        errors="coerce"
    )

    df["year"] = df["inverse_data"].dt.year

    df["month"] = df["inverse_data"].dt.month

    df["month_name"] = (
        df["inverse_data"]
        .dt.month_name()
    )

    print("\nDate column processed successfully.")

# ============================================================
# 12. TIME PROCESSING
# ============================================================

if "hrmn" in df.columns:

    # Convert to string first
    time_text = (
        df["hrmn"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    # Keep only numbers
    time_text = (
        time_text
        .str.replace(r"\D", "", regex=True)
    )

    # Make 4-digit time
    time_text = time_text.str.zfill(4)

    df["hour"] = pd.to_numeric(
        time_text.str[:2],
        errors="coerce"
    )

    df["minute"] = pd.to_numeric(
        time_text.str[2:4],
        errors="coerce"
    )

    # Keep valid hours only
    df.loc[
        ~df["hour"].between(0, 23),
        "hour"
    ] = np.nan

    print("\nTime column processed successfully.")

# ============================================================
# 13. TIME PERIOD
# ============================================================

if "hour" in df.columns:

    def get_time_period(hour):

        if pd.isna(hour):
            return "Unknown"

        if hour < 6:
            return "Night"

        elif hour < 12:
            return "Morning"

        elif hour < 17:
            return "Afternoon"

        elif hour < 21:
            return "Evening"

        else:
            return "Night"

    df["time_period"] = (
        df["hour"]
        .apply(get_time_period)
    )

# ============================================================
# 14. DRIVER AGE CLEANING
# ============================================================

if "driver_age" in df.columns:

    df["driver_age"] = pd.to_numeric(
        df["driver_age"],
        errors="coerce"
    )

# ============================================================
# 15. CAR AGE CLEANING
# ============================================================

if "car_age" in df.columns:

    df["car_age"] = pd.to_numeric(
        df["car_age"],
        errors="coerce"
    )

# ============================================================
# 16. ENGINE SIZE CLEANING
# ============================================================

if "engine_size" in df.columns:

    df["engine_size"] = pd.to_numeric(
        df["engine_size"],
        errors="coerce"
    )

# ============================================================
# 17. CASUALTY AGE CLEANING
# ============================================================

if "casualty_age" in df.columns:

    df["casualty_age"] = pd.to_numeric(
        df["casualty_age"],
        errors="coerce"
    )

# ============================================================
# 18. ACCIDENT SEVERITY
# ============================================================

if "severity" in df.columns:

    print("\n" + "=" * 70)
    print("ACCIDENT SEVERITY ANALYSIS")
    print("=" * 70)

    severity_count = (
        df["severity"]
        .value_counts()
    )

    print(
        severity_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    severity_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accident Severity Distribution",
        fontsize=16
    )

    ax.set_xlabel(
        "Accident Severity"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 19. WEATHER ANALYSIS
# ============================================================

if "weather" in df.columns:

    print("\n" + "=" * 70)
    print("WEATHER ANALYSIS")
    print("=" * 70)

    weather_count = (
        df["weather"]
        .value_counts()
    )

    print(
        weather_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True
    )

    weather_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Weather Condition",
        fontsize=16
    )

    ax.set_xlabel(
        "Weather Condition"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    plt.show()

# ============================================================
# 20. LIGHTING CONDITION
# ============================================================

if "lum" in df.columns:

    print("\n" + "=" * 70)
    print("LIGHTING CONDITION ANALYSIS")
    print("=" * 70)

    lighting_count = (
        df["lum"]
        .value_counts()
    )

    print(
        lighting_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    lighting_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Lighting Condition",
        fontsize=16
    )

    ax.set_xlabel(
        "Lighting Condition"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 21. TIME OF DAY
# ============================================================

if "hour" in df.columns:

    print("\n" + "=" * 70)
    print("TIME OF DAY ANALYSIS")
    print("=" * 70)

    hour_count = (
        df["hour"]
        .value_counts()
        .sort_index()
    )

    print(
        hour_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(13, 6),
        constrained_layout=True
    )

    ax.plot(
        hour_count.index,
        hour_count.values,
        marker="o"
    )

    ax.set_title(
        "Accidents by Hour of the Day",
        fontsize=16
    )

    ax.set_xlabel(
        "Hour of Day"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.set_xticks(
        range(0, 24)
    )

    ax.grid(True)

    plt.show()

# ============================================================
# 22. TIME PERIOD
# ============================================================

if "time_period" in df.columns:

    print("\n" + "=" * 70)
    print("TIME PERIOD ANALYSIS")
    print("=" * 70)

    period_order = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]

    period_count = (
        df["time_period"]
        .value_counts()
        .reindex(
            period_order,
            fill_value=0
        )
    )

    print(
        period_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    period_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Time Period",
        fontsize=16
    )

    ax.set_xlabel(
        "Time Period"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 23. DAY OF WEEK
# ============================================================

if "week_day" in df.columns:

    print("\n" + "=" * 70)
    print("DAY OF WEEK ANALYSIS")
    print("=" * 70)

    day_count = (
        df["week_day"]
        .value_counts()
    )

    print(
        day_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    day_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Day of Week",
        fontsize=16
    )

    ax.set_xlabel(
        "Day"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 24. STATE-WISE ACCIDENTS
# ============================================================

if "state" in df.columns:

    print("\n" + "=" * 70)
    print("STATE-WISE ACCIDENT ANALYSIS")
    print("=" * 70)

    state_count = (
        df["state"]
        .value_counts()
    )

    print(
        state_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True
    )

    state_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Indian State",
        fontsize=16
    )

    ax.set_xlabel(
        "State"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.show()

# ============================================================
# 25. TOP ACCIDENT LOCATIONS
# ============================================================

if "location" in df.columns:

    print("\n" + "=" * 70)
    print("ACCIDENT HOTSPOT / LOCATION ANALYSIS")
    print("=" * 70)

    location_count = (
        df["location"]
        .value_counts()
        .head(15)
    )

    print(
        location_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    location_count.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_title(
        "Top Accident Locations",
        fontsize=16
    )

    ax.set_xlabel(
        "Number of Accidents"
    )

    ax.set_ylabel(
        "Location"
    )

    plt.show()

# ============================================================
# 26. VEHICLE TYPE
# ============================================================

if "vehicle_type" in df.columns:

    print("\n" + "=" * 70)
    print("VEHICLE TYPE ANALYSIS")
    print("=" * 70)

    vehicle_count = (
        df["vehicle_type"]
        .value_counts()
    )

    print(
        vehicle_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True
    )

    vehicle_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Vehicle Type",
        fontsize=16
    )

    ax.set_xlabel(
        "Vehicle Type"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    plt.show()

# ============================================================
# 27. DRIVER AGE DISTRIBUTION
# ============================================================

if "driver_age" in df.columns:

    print("\n" + "=" * 70)
    print("DRIVER AGE ANALYSIS")
    print("=" * 70)

    print(
        df["driver_age"].describe()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True
    )

    ax.hist(
        df["driver_age"].dropna(),
        bins=20
    )

    ax.set_title(
        "Distribution of Driver Age",
        fontsize=16
    )

    ax.set_xlabel(
        "Driver Age"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    plt.show()

# ============================================================
# 28. DRIVER AGE GROUP
# ============================================================

if "driver_age" in df.columns:

    bins = [
        0,
        18,
        25,
        35,
        45,
        60,
        100
    ]

    labels = [
        "Below 18",
        "18-24",
        "25-34",
        "35-44",
        "45-59",
        "60+"
    ]

    df["driver_age_group"] = pd.cut(
        df["driver_age"],
        bins=bins,
        labels=labels,
        right=False
    )

    age_group_count = (
        df["driver_age_group"]
        .value_counts()
        .reindex(labels)
    )

    print("\n" + "=" * 70)
    print("DRIVER AGE GROUP ANALYSIS")
    print("=" * 70)

    print(
        age_group_count.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True
    )

    age_group_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Driver Age Group",
        fontsize=16
    )

    ax.set_xlabel(
        "Driver Age Group"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 29. DRIVER GENDER
# ============================================================

if "driver_sex" in df.columns:

    print("\n" + "=" * 70)
    print("DRIVER GENDER ANALYSIS")
    print("=" * 70)

    driver_gender = (
        df["driver_sex"]
        .value_counts()
    )

    print(
        driver_gender.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(8, 6),
        constrained_layout=True
    )

    ax.pie(
        driver_gender.values,
        labels=driver_gender.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Accidents by Driver Gender",
        fontsize=16
    )

    plt.show()

# ============================================================
# 30. CASUALTY SEVERITY
# ============================================================

if "casualty_severity" in df.columns:

    print("\n" + "=" * 70)
    print("CASUALTY SEVERITY ANALYSIS")
    print("=" * 70)

    casualty_severity = (
        df["casualty_severity"]
        .value_counts()
    )

    print(
        casualty_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    casualty_severity.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Casualty Severity Distribution",
        fontsize=16
    )

    ax.set_xlabel(
        "Casualty Severity"
    )

    ax.set_ylabel(
        "Number of Casualties"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 31. CASUALTY TYPE
# ============================================================

if "casualty_type" in df.columns:

    print("\n" + "=" * 70)
    print("CASUALTY TYPE ANALYSIS")
    print("=" * 70)

    casualty_type = (
        df["casualty_type"]
        .value_counts()
    )

    print(
        casualty_type.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True
    )

    casualty_type.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Accidents by Casualty Type",
        fontsize=16
    )

    ax.set_xlabel(
        "Casualty Type"
    )

    ax.set_ylabel(
        "Number of Casualties"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    plt.show()

# ============================================================
# 32. WEATHER VS SEVERITY
# ============================================================

if (
    "weather" in df.columns
    and "severity" in df.columns
):

    weather_severity = pd.crosstab(
        df["weather"],
        df["severity"]
    )

    print("\n" + "=" * 70)
    print("WEATHER VS ACCIDENT SEVERITY")
    print("=" * 70)

    print(
        weather_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    sns.heatmap(
        weather_severity,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        "Weather vs Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "Accident Severity"
    )

    ax.set_ylabel(
        "Weather"
    )

    plt.show()

# ============================================================
# 33. LIGHTING VS SEVERITY
# ============================================================

if (
    "lum" in df.columns
    and "severity" in df.columns
):

    lighting_severity = pd.crosstab(
        df["lum"],
        df["severity"]
    )

    print("\n" + "=" * 70)
    print("LIGHTING VS ACCIDENT SEVERITY")
    print("=" * 70)

    print(
        lighting_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 7),
        constrained_layout=True
    )

    sns.heatmap(
        lighting_severity,
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=ax
    )

    ax.set_title(
        "Lighting Condition vs Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "Accident Severity"
    )

    ax.set_ylabel(
        "Lighting Condition"
    )

    plt.show()

# ============================================================
# 34. VEHICLE TYPE VS SEVERITY
# ============================================================

if (
    "vehicle_type" in df.columns
    and "severity" in df.columns
):

    vehicle_severity = pd.crosstab(
        df["vehicle_type"],
        df["severity"]
    )

    print("\n" + "=" * 70)
    print("VEHICLE TYPE VS SEVERITY")
    print("=" * 70)

    print(
        vehicle_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    vehicle_severity.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Vehicle Type vs Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "Vehicle Type"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    ax.legend(
        title="Severity"
    )

    plt.show()

# ============================================================
# 35. DRIVER GENDER VS SEVERITY
# ============================================================

if (
    "driver_sex" in df.columns
    and "severity" in df.columns
):

    gender_severity = pd.crosstab(
        df["driver_sex"],
        df["severity"]
    )

    print("\n" + "=" * 70)
    print("DRIVER GENDER VS SEVERITY")
    print("=" * 70)

    print(
        gender_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    gender_severity.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Driver Gender vs Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "Driver Gender"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Severity"
    )

    plt.show()

# ============================================================
# 36. TIME PERIOD VS SEVERITY
# ============================================================

if (
    "time_period" in df.columns
    and "severity" in df.columns
):

    period_severity = pd.crosstab(
        df["time_period"],
        df["severity"]
    )

    period_order = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]

    period_severity = (
        period_severity
        .reindex(period_order)
        .fillna(0)
    )

    print("\n" + "=" * 70)
    print("TIME PERIOD VS SEVERITY")
    print("=" * 70)

    print(
        period_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7),
        constrained_layout=True
    )

    period_severity.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Time Period vs Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "Time Period"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Severity"
    )

    plt.show()

# ============================================================
# 37. STATE VS SEVERITY
# ============================================================

if (
    "state" in df.columns
    and "severity" in df.columns
):

    state_severity = pd.crosstab(
        df["state"],
        df["severity"]
    )

    print("\n" + "=" * 70)
    print("STATE VS SEVERITY")
    print("=" * 70)

    print(
        state_severity.to_string()
    )

    # Use top 15 states by total accidents
    top_states = (
        df["state"]
        .value_counts()
        .head(15)
        .index
    )

    state_severity_top = (
        state_severity
        .loc[
            state_severity.index.isin(
                top_states
            )
        ]
    )

    fig, ax = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True
    )

    state_severity_top.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Top States: Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "State"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.legend(
        title="Severity"
    )

    plt.show()

# ============================================================
# 38. MONTHLY ACCIDENT TREND
# ============================================================

if "inverse_data" in df.columns:

    monthly = (
        df
        .dropna(subset=["inverse_data"])
        .groupby(
            df["inverse_data"].dt.to_period("M")
        )
        .size()
    )

    print("\n" + "=" * 70)
    print("MONTHLY ACCIDENT TREND")
    print("=" * 70)

    print(
        monthly.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(14, 6),
        constrained_layout=True
    )

    ax.plot(
        monthly.index.astype(str),
        monthly.values,
        marker="o"
    )

    ax.set_title(
        "Monthly Accident Trend",
        fontsize=16
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    ax.tick_params(
        axis="x",
        rotation=90
    )

    ax.grid(True)

    plt.show()

# ============================================================
# 39. DRIVER AGE VS SEVERITY
# ============================================================

if (
    "driver_age" in df.columns
    and "severity" in df.columns
):

    age_severity = (
        df.groupby("severity")["driver_age"]
        .mean()
        .sort_values()
    )

    print("\n" + "=" * 70)
    print("AVERAGE DRIVER AGE BY SEVERITY")
    print("=" * 70)

    print(
        age_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    age_severity.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Driver Age by Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "Accident Severity"
    )

    ax.set_ylabel(
        "Average Driver Age"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 40. CAR AGE VS SEVERITY
# ============================================================

if (
    "car_age" in df.columns
    and "severity" in df.columns
):

    car_age_severity = (
        df.groupby("severity")["car_age"]
        .mean()
        .sort_values()
    )

    print("\n" + "=" * 70)
    print("AVERAGE CAR AGE BY SEVERITY")
    print("=" * 70)

    print(
        car_age_severity.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    car_age_severity.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Vehicle Age by Accident Severity",
        fontsize=16
    )

    ax.set_xlabel(
        "Accident Severity"
    )

    ax.set_ylabel(
        "Average Vehicle Age"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.show()

# ============================================================
# 41. CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

if len(numeric_df.columns) >= 2:

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    correlation = (
        numeric_df.corr()
    )

    print(
        correlation.to_string()
    )

    fig, ax = plt.subplots(
        figsize=(12, 9),
        constrained_layout=True
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=16
    )

    plt.show()

# ============================================================
# 42. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("                    FINAL SUMMARY")
print("=" * 70)

print(
    "\nTotal accident records analyzed:",
    len(df)
)

print(
    "Total columns:",
    len(df.columns)
)

if "severity" in df.columns:

    print("\nAccident severity:")

    print(
        df["severity"]
        .value_counts()
        .to_string()
    )

if "weather" in df.columns:

    print("\nWeather:")

    print(
        df["weather"]
        .value_counts()
        .to_string()
    )

if "lum" in df.columns:

    print("\nLighting:")

    print(
        df["lum"]
        .value_counts()
        .to_string()
    )

if "vehicle_type" in df.columns:

    print("\nVehicle type:")

    print(
        df["vehicle_type"]
        .value_counts()
        .to_string()
    )

if "state" in df.columns:

    print("\nTop 10 states:")

    print(
        df["state"]
        .value_counts()
        .head(10)
        .to_string()
    )

if "time_period" in df.columns:

    print("\nTime period:")

    print(
        df["time_period"]
        .value_counts()
        .to_string()
    )

if "casualty_severity" in df.columns:

    print("\nCasualty severity:")

    print(
        df["casualty_severity"]
        .value_counts()
        .to_string()
    )

print("\n")
print("=" * 70)
print("       TASK-05 ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

input("\nPress Enter to close...")
