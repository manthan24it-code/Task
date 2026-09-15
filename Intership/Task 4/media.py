# ============================================================
# TASK-04
# SOCIAL MEDIA SENTIMENT ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "sentimentdataset.csv")

print("==========================================")
print("       SOCIAL MEDIA SENTIMENT ANALYSIS")
print("==========================================")

print("\nLoading dataset...")
print("File:", file_path)

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# ============================================================
# 3. DISPLAY ALL RECORDS
# ============================================================

print("\n========== ALL RECORDS ==========")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(df)

# ============================================================
# 4. FIRST 10 RECORDS
# ============================================================

print("\n========== FIRST 10 RECORDS ==========")
print(df.head(10))

# ============================================================
# 5. LAST 10 RECORDS
# ============================================================

print("\n========== LAST 10 RECORDS ==========")
print(df.tail(10))

# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

df.info()

# ============================================================
# 7. STATISTICAL SUMMARY
# ============================================================

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe(include="all"))

# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

# ============================================================
# 9. DUPLICATE RECORDS
# ============================================================

print("\n========== DUPLICATE RECORDS ==========")

duplicates = df.duplicated().sum()

print("Duplicate Records:", duplicates)

# ============================================================
# 10. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("Records after removing duplicates:", len(df))

# ============================================================
# 11. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Likes",
    "Retweets",
    "Year",
    "Month",
    "Day",
    "Hour"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ============================================================
# 12. SENTIMENT ANALYSIS
# ============================================================

if "Sentiment" in df.columns:

    print("\n==========================================")
    print("          SENTIMENT ANALYSIS")
    print("==========================================")

    sentiment_count = df["Sentiment"].value_counts()

    print("\nSentiment Count:")
    print(sentiment_count)

    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    sentiment_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Distribution of Social Media Sentiments",
        fontsize=16
    )

    ax.set_xlabel(
        "Sentiment",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    plt.show()

    # --------------------------------------------------------
    # PIE CHART
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 9),
        constrained_layout=True
    )

    ax.pie(
        sentiment_count.values,
        labels=sentiment_count.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Percentage Distribution of Sentiments",
        fontsize=16
    )

    plt.show()

# ============================================================
# 13. PLATFORM ANALYSIS
# ============================================================

if "Platform" in df.columns:

    print("\n==========================================")
    print("           PLATFORM ANALYSIS")
    print("==========================================")

    platform_count = df["Platform"].value_counts()

    print("\nPosts by Platform:")
    print(platform_count)

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    platform_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Number of Posts by Social Media Platform",
        fontsize=16
    )

    ax.set_xlabel(
        "Platform",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    plt.show()

# ============================================================
# 14. SENTIMENT BY PLATFORM
# ============================================================

if "Platform" in df.columns and "Sentiment" in df.columns:

    print("\n==========================================")
    print("        SENTIMENT BY PLATFORM")
    print("==========================================")

    platform_sentiment = pd.crosstab(
        df["Platform"],
        df["Sentiment"]
    )

    print(platform_sentiment)

    fig, ax = plt.subplots(
        figsize=(14, 8),
        constrained_layout=True
    )

    platform_sentiment.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Sentiment Distribution Across Platforms",
        fontsize=16
    )

    ax.set_xlabel(
        "Platform",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    ax.legend(
        title="Sentiment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.show()

# ============================================================
# 15. HASHTAG ANALYSIS
# ============================================================

if "Hashtags" in df.columns:

    print("\n==========================================")
    print("           HASHTAG ANALYSIS")
    print("==========================================")

    hashtag_count = df["Hashtags"].value_counts()

    print("\nTop 10 Hashtags:")
    print(hashtag_count.head(10))

    top_hashtags = hashtag_count.head(10)

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    top_hashtags.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_title(
        "Top 10 Most Used Hashtags",
        fontsize=16
    )

    ax.set_xlabel(
        "Number of Posts",
        fontsize=12
    )

    ax.set_ylabel(
        "Hashtag",
        fontsize=12
    )

    plt.show()

# ============================================================
# 16. LIKES ANALYSIS
# ============================================================

if "Likes" in df.columns:

    print("\n==========================================")
    print("             LIKES ANALYSIS")
    print("==========================================")

    print(df["Likes"].describe())

    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    ax.hist(
        df["Likes"].dropna(),
        bins=30
    )

    ax.set_title(
        "Distribution of Likes",
        fontsize=16
    )

    ax.set_xlabel(
        "Number of Likes",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    plt.show()

# ============================================================
# 17. RETWEETS ANALYSIS
# ============================================================

if "Retweets" in df.columns:

    print("\n==========================================")
    print("           RETWEETS ANALYSIS")
    print("==========================================")

    print(df["Retweets"].describe())

    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    ax.hist(
        df["Retweets"].dropna(),
        bins=30
    )

    ax.set_title(
        "Distribution of Retweets",
        fontsize=16
    )

    ax.set_xlabel(
        "Number of Retweets",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    plt.show()

# ============================================================
# 18. SENTIMENT VS LIKES
# ============================================================

if "Sentiment" in df.columns and "Likes" in df.columns:

    fig, ax = plt.subplots(
        figsize=(14, 8),
        constrained_layout=True
    )

    sns.boxplot(
        data=df,
        x="Sentiment",
        y="Likes",
        ax=ax
    )

    ax.set_title(
        "Likes Distribution by Sentiment",
        fontsize=16
    )

    ax.set_xlabel(
        "Sentiment",
        fontsize=12
    )

    ax.set_ylabel(
        "Likes",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    plt.show()

# ============================================================
# 19. SENTIMENT VS RETWEETS
# ============================================================

if "Sentiment" in df.columns and "Retweets" in df.columns:

    fig, ax = plt.subplots(
        figsize=(14, 8),
        constrained_layout=True
    )

    sns.boxplot(
        data=df,
        x="Sentiment",
        y="Retweets",
        ax=ax
    )

    ax.set_title(
        "Retweets Distribution by Sentiment",
        fontsize=16
    )

    ax.set_xlabel(
        "Sentiment",
        fontsize=12
    )

    ax.set_ylabel(
        "Retweets",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    plt.show()

# ============================================================
# 20. COUNTRY ANALYSIS
# ============================================================

if "Country" in df.columns:

    print("\n==========================================")
    print("            COUNTRY ANALYSIS")
    print("==========================================")

    country_count = df["Country"].value_counts()

    print("\nTop 10 Countries:")
    print(country_count.head(10))

    top_countries = country_count.head(10)

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    top_countries.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_title(
        "Top 10 Countries by Number of Posts",
        fontsize=16
    )

    ax.set_xlabel(
        "Number of Posts",
        fontsize=12
    )

    ax.set_ylabel(
        "Country",
        fontsize=12
    )

    plt.show()

# ============================================================
# 21. YEAR-WISE SENTIMENT
# ============================================================

if "Year" in df.columns and "Sentiment" in df.columns:

    print("\n==========================================")
    print("          YEAR-WISE SENTIMENT")
    print("==========================================")

    year_sentiment = pd.crosstab(
        df["Year"],
        df["Sentiment"]
    )

    print(year_sentiment)

    fig, ax = plt.subplots(
        figsize=(14, 8),
        constrained_layout=True
    )

    year_sentiment.plot(
        kind="line",
        marker="o",
        ax=ax
    )

    ax.set_title(
        "Sentiment Trends Over the Years",
        fontsize=16
    )

    ax.set_xlabel(
        "Year",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    ax.legend(
        title="Sentiment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.show()

# ============================================================
# 22. MONTH-WISE SENTIMENT
# ============================================================

if "Month" in df.columns and "Sentiment" in df.columns:

    print("\n==========================================")
    print("          MONTH-WISE SENTIMENT")
    print("==========================================")

    month_sentiment = pd.crosstab(
        df["Month"],
        df["Sentiment"]
    )

    print(month_sentiment)

    fig, ax = plt.subplots(
        figsize=(14, 8),
        constrained_layout=True
    )

    month_sentiment.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Monthly Sentiment Distribution",
        fontsize=16
    )

    ax.set_xlabel(
        "Month",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    ax.legend(
        title="Sentiment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.show()

# ============================================================
# 23. DAY-WISE ANALYSIS
# ============================================================

if "Day" in df.columns:

    print("\n==========================================")
    print("           DAY-WISE ANALYSIS")
    print("==========================================")

    day_count = df["Day"].value_counts().sort_index()

    print(day_count)

    fig, ax = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True
    )

    day_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Number of Posts by Day",
        fontsize=16
    )

    ax.set_xlabel(
        "Day",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    plt.show()

# ============================================================
# 24. HOUR-WISE ANALYSIS
# ============================================================

if "Hour" in df.columns:

    print("\n==========================================")
    print("           HOUR-WISE ANALYSIS")
    print("==========================================")

    hour_count = df["Hour"].value_counts().sort_index()

    print(hour_count)

    fig, ax = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True
    )

    hour_count.plot(
        kind="line",
        marker="o",
        ax=ax
    )

    ax.set_title(
        "Social Media Activity by Hour",
        fontsize=16
    )

    ax.set_xlabel(
        "Hour",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Posts",
        fontsize=12
    )

    ax.grid(True)

    plt.show()

# ============================================================
# 25. CORRELATION HEATMAP
# ============================================================

print("\n==========================================")
print("          CORRELATION ANALYSIS")
print("==========================================")

numeric_data = df.select_dtypes(
    include="number"
)

print("\nNumeric Columns:")
print(numeric_data.columns.tolist())

if len(numeric_data.columns) >= 2:

    correlation = numeric_data.corr()

    print("\nCorrelation Matrix:")
    print(correlation)

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
# 26. TOP 10 MOST LIKED POSTS
# ============================================================

if "Likes" in df.columns and "Text" in df.columns:

    print("\n==========================================")
    print("        TOP 10 MOST LIKED POSTS")
    print("==========================================")

    top_liked = df.sort_values(
        by="Likes",
        ascending=False
    ).head(10)

    print(
        top_liked[
            ["Text", "Sentiment", "Likes"]
        ].to_string(index=False)
    )

# ============================================================
# 27. TOP 10 MOST RETWEETED POSTS
# ============================================================

if "Retweets" in df.columns and "Text" in df.columns:

    print("\n==========================================")
    print("       TOP 10 MOST RETWEETED POSTS")
    print("==========================================")

    top_retweets = df.sort_values(
        by="Retweets",
        ascending=False
    ).head(10)

    print(
        top_retweets[
            ["Text", "Sentiment", "Retweets"]
        ].to_string(index=False)
    )

# ============================================================
# 28. AVERAGE LIKES BY SENTIMENT
# ============================================================

if "Sentiment" in df.columns and "Likes" in df.columns:

    print("\n==========================================")
    print("       AVERAGE LIKES BY SENTIMENT")
    print("==========================================")

    average_likes = df.groupby(
        "Sentiment"
    )["Likes"].mean().sort_values(
        ascending=False
    )

    print(average_likes)

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    average_likes.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Likes by Sentiment",
        fontsize=16
    )

    ax.set_xlabel(
        "Sentiment",
        fontsize=12
    )

    ax.set_ylabel(
        "Average Likes",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    plt.show()

# ============================================================
# 29. AVERAGE RETWEETS BY SENTIMENT
# ============================================================

if "Sentiment" in df.columns and "Retweets" in df.columns:

    print("\n==========================================")
    print("      AVERAGE RETWEETS BY SENTIMENT")
    print("==========================================")

    average_retweets = df.groupby(
        "Sentiment"
    )["Retweets"].mean().sort_values(
        ascending=False
    )

    print(average_retweets)

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True
    )

    average_retweets.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Retweets by Sentiment",
        fontsize=16
    )

    ax.set_xlabel(
        "Sentiment",
        fontsize=12
    )

    ax.set_ylabel(
        "Average Retweets",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    plt.show()

# ============================================================
# 30. AVERAGE ENGAGEMENT BY SENTIMENT
# ============================================================

if (
    "Sentiment" in df.columns
    and "Likes" in df.columns
    and "Retweets" in df.columns
):

    print("\n==========================================")
    print("       AVERAGE SOCIAL MEDIA ENGAGEMENT")
    print("==========================================")

    engagement = df.groupby(
        "Sentiment"
    )[["Likes", "Retweets"]].mean()

    print(engagement)

    fig, ax = plt.subplots(
        figsize=(14, 8),
        constrained_layout=True
    )

    engagement.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Engagement by Sentiment",
        fontsize=16
    )

    ax.set_xlabel(
        "Sentiment",
        fontsize=12
    )

    ax.set_ylabel(
        "Average Count",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=45)

    ax.legend(
        title="Engagement",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.show()

# ============================================================
# 31. SENTIMENT PERCENTAGE
# ============================================================

if "Sentiment" in df.columns:

    print("\n==========================================")
    print("         SENTIMENT PERCENTAGE")
    print("==========================================")

    percentage = (
        df["Sentiment"]
        .value_counts(normalize=True)
        * 100
    )

    print(
        percentage.round(2)
    )

# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n")
print("==============================================")
print("              FINAL SUMMARY")
print("==============================================")

print("\nTotal Records Analyzed:")
print(len(df))

print("\nTotal Columns:")
print(len(df.columns))

if "Sentiment" in df.columns:

    print("\nSentiment Distribution:")
    print(df["Sentiment"].value_counts())

if "Platform" in df.columns:

    print("\nPlatform Distribution:")
    print(df["Platform"].value_counts())

if "Country" in df.columns:

    print("\nTop 10 Countries:")
    print(
        df["Country"]
        .value_counts()
        .head(10)
    )

if "Hashtags" in df.columns:

    print("\nTop 10 Hashtags:")
    print(
        df["Hashtags"]
        .value_counts()
        .head(10)
    )

if "Likes" in df.columns:

    print("\nAverage Likes:")
    print(
        df["Likes"].mean()
    )

if "Retweets" in df.columns:

    print("\nAverage Retweets:")
    print(
        df["Retweets"].mean()
    )

print("\n==============================================")
print("       SOCIAL MEDIA ANALYSIS COMPLETED")
print("==============================================")
