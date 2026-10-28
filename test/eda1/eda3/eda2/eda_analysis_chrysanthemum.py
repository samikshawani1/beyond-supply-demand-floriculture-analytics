# ============================================================
# CHRYSANTHEMUM PRICE EDA
# Period: 2010-2025
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path(
    r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\all_data.csv"
)

# The script should be executed from your project's eda folder.
OUTPUT_DIR = Path.cwd() / "eda_outputs"

PLOTS_DIR = OUTPUT_DIR / "plots"
YEARLY_MONTHLY_DIR = PLOTS_DIR / "monthly_average_by_year"
YEARLY_EVENTS_DIR = PLOTS_DIR / "yearly_event_analysis"

REPORTS_DIR = OUTPUT_DIR / "reports"

for folder in [
    OUTPUT_DIR,
    PLOTS_DIR,
    YEARLY_MONTHLY_DIR,
    YEARLY_EVENTS_DIR,
    REPORTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# Change this if your price has a known unit.
PRICE_UNIT = "Recorded price (confirm market unit)"

# IQR multiplier used for statistical outlier detection.
IQR_MULTIPLIER = 1.5

# Optional: add verified auspicious-day names or keywords here.
# These must match the actual labels in your Events column.
#
# Do not add generic terms such as "Purnima" or "Ekadashi"
# unless your project methodology explicitly defines them as
# auspicious days.

AUSPICIOUS_KEYWORDS = [
    # Example:
    # "Vivah Muhurat",
    # "Marriage Muhurat",
]

# Optional: add event labels that are confirmed festivals.
# Festival keywords should be based on your verified calendar.

FESTIVAL_KEYWORDS = [
    "festival",
    "puja",
    "jayanti",
    "janmotsava",
    "navratri",
    "shivaratri",
    "chaturthi",
    "rath yatra",
    "rathyatra",
    "janmashtami",
    "diwali",
    "deepavali",
    "holi",
    "dussehra",
    "vijayadashami",
    "ugadi",
    "gudi padwa",
    "sankranti",
    "pongal",
    "onam",
    "raksha bandhan",
    "rakshabandhan",
    "karwa chauth",
    "teej",
    "chhath",
    "ganesh",
    "hanuman",
    "rama navami",
    "ram navami",
    "krishna janmashtami",
    "makar sankranti",
]

# ============================================================
# 2. GENERAL PLOTTING STYLE
# ============================================================

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_plot(filename):
    """Save the current plot and close the figure."""

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def clean_filename(text):
    """Convert a label into a safe filename."""

    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text))


def format_date_axis(ax):
    """Make date-axis labels readable."""

    ax.xaxis.set_major_locator(
        mdates.AutoDateLocator(minticks=5, maxticks=12)
    )

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            ax.xaxis.get_major_locator()
        )
    )

    ax.tick_params(axis="x", rotation=0)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("CHRYSANTHEMUM PRICE EDA")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please verify the CSV path."
    )

df = pd.read_csv(DATA_PATH)

# Standardise column names.
df.columns = df.columns.str.strip()

required_columns = [
    "Date",
    "Price",
    "Tithi",
    "Masa",
    "Events"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Convert data types.
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

# Trim text values.
for col in ["Tithi", "Masa", "Events"]:
    df[col] = df[col].astype("string").str.strip()

    df[col] = df[col].replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA
    })

# Report invalid values before excluding unusable rows.
invalid_date_count = int(df["Date"].isna().sum())
invalid_price_count = int(df["Price"].isna().sum())

# A price-date observation cannot be analysed without both fields.
df = df.dropna(subset=["Date", "Price"]).copy()

df = df.sort_values("Date").reset_index(drop=True)

# Calendar features.
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")
df["Day_Of_Week"] = df["Date"].dt.day_name()
df["Day_Of_Week_Number"] = df["Date"].dt.dayofweek
df["Day_Of_Year"] = df["Date"].dt.dayofyear

# ============================================================
# 4. DATASET OVERVIEW
# ============================================================

print("\n--- DATASET OVERVIEW ---")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("Date range:", df["Date"].min(), "to", df["Date"].max())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst five records:")
print(df.head())

print("\nLast five records:")
print(df.tail())

print("\nDescriptive statistics:")
print(df.describe(include="all").to_string())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate dates:", df["Date"].duplicated().sum())

# Save the cleaned analytical dataset.
df.to_csv(
    REPORTS_DIR / "cleaned_eda_dataset.csv",
    index=False
)

# ============================================================
# 5. DATA QUALITY REPORT
# ============================================================

quality_report = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing_Count": [df[c].isna().sum() for c in df.columns],
    "Missing_Percentage": [
        df[c].isna().mean() * 100
        for c in df.columns
    ],
    "Unique_Values": [df[c].nunique(dropna=True) for c in df.columns]
})

quality_report.to_csv(
    REPORTS_DIR / "data_quality_report.csv",
    index=False
)

print("\nData quality report saved.")

# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

price_stats = df["Price"].describe()

q1 = df["Price"].quantile(0.25)
q3 = df["Price"].quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - IQR_MULTIPLIER * iqr
upper_fence = q3 + IQR_MULTIPLIER * iqr

summary = pd.DataFrame({
    "Metric": [
        "Number of observations",
        "Minimum",
        "Maximum",
        "Mean",
        "Median",
        "Standard deviation",
        "Variance",
        "Q1",
        "Q3",
        "IQR",
        "Lower IQR fence",
        "Upper IQR fence",
        "Skewness",
        "Kurtosis"
    ],
    "Value": [
        len(df),
        df["Price"].min(),
        df["Price"].max(),
        df["Price"].mean(),
        df["Price"].median(),
        df["Price"].std(),
        df["Price"].var(),
        q1,
        q3,
        iqr,
        lower_fence,
        upper_fence,
        df["Price"].skew(),
        df["Price"].kurt()
    ]
})

summary.to_csv(
    REPORTS_DIR / "price_descriptive_statistics.csv",
    index=False
)

print("\n--- PRICE SUMMARY ---")
print(summary.to_string(index=False))

# ============================================================
# 7. WHOLE-DATASET OUTLIER DETECTION
# ============================================================

df["Is_Outlier"] = (
    (df["Price"] < lower_fence) |
    (df["Price"] > upper_fence)
)

outliers = df.loc[df["Is_Outlier"]].copy()

print("\n--- OUTLIER ANALYSIS ---")
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower fence:", lower_fence)
print("Upper fence:", upper_fence)
print("Total outliers:", len(outliers))
print(
    "Outlier percentage:",
    round(df["Is_Outlier"].mean() * 100, 2),
    "%"
)

outliers.to_csv(
    REPORTS_DIR / "whole_dataset_outliers.csv",
    index=False
)

# Graph 1: Whole-dataset boxplot.
plt.figure(figsize=(12, 5))

sns.boxplot(
    x=df["Price"],
    showmeans=True
)

plt.title("Chrysanthemum Prices: Whole-Dataset Outlier Analysis")
plt.xlabel(PRICE_UNIT)
plt.ylabel("Distribution")

save_plot(PLOTS_DIR / "01_whole_dataset_boxplot.png")

# Graph 2: Whole-dataset scatter plot highlighting outliers.
fig, ax = plt.subplots(figsize=(15, 6))

ax.scatter(
    df["Date"],
    df["Price"],
    s=12,
    alpha=0.45,
    label="Regular observations"
)

if not outliers.empty:
    ax.scatter(
        outliers["Date"],
        outliers["Price"],
        s=35,
        marker="x",
        label="IQR outliers"
    )

ax.set_title("Chrysanthemum Prices Over Time: Outliers Highlighted")
ax.set_xlabel("Date")
ax.set_ylabel(PRICE_UNIT)

format_date_axis(ax)
ax.legend()

save_plot(PLOTS_DIR / "02_whole_dataset_outliers_over_time.png")

# Graph 3: Price distribution.
fig, ax = plt.subplots(figsize=(11, 6))

sns.histplot(
    data=df,
    x="Price",
    bins=40,
    kde=True,
    ax=ax
)

ax.set_title("Distribution of Daily Chrysanthemum Prices")
ax.set_xlabel(PRICE_UNIT)
ax.set_ylabel("Number of Observations")

save_plot(PLOTS_DIR / "03_price_distribution.png")

# Graph 4: Log-scale price distribution, useful for right-skewed data.
positive_prices = df.loc[df["Price"] > 0, "Price"]

if not positive_prices.empty:
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.histplot(
        x=np.log1p(positive_prices),
        bins=40,
        kde=True,
        ax=ax
    )

    ax.set_title("Log-Transformed Chrysanthemum Price Distribution")
    ax.set_xlabel("log(1 + Price)")
    ax.set_ylabel("Number of Observations")

    save_plot(PLOTS_DIR / "04_log_price_distribution.png")

# ============================================================
# 8. YEAR-WISE PRICE TRENDS
# ============================================================

yearly_stats = df.groupby("Year").agg(
    Observation_Count=("Price", "count"),
    Mean_Price=("Price", "mean"),
    Median_Price=("Price", "median"),
    Minimum_Price=("Price", "min"),
    Maximum_Price=("Price", "max"),
    Standard_Deviation=("Price", "std"),
    Price_Variance=("Price", "var")
).reset_index()

yearly_stats["YoY_Mean_Change_Percent"] = (
    yearly_stats["Mean_Price"].pct_change() * 100
)

yearly_stats.to_csv(
    REPORTS_DIR / "yearly_price_statistics.csv",
    index=False
)

# Graph 5: Annual mean and median prices.
fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    yearly_stats["Year"],
    yearly_stats["Mean_Price"],
    marker="o",
    label="Mean price"
)

ax.plot(
    yearly_stats["Year"],
    yearly_stats["Median_Price"],
    marker="s",
    label="Median price"
)

ax.set_title("Annual Mean and Median Chrysanthemum Prices")
ax.set_xlabel("Year")
ax.set_ylabel(PRICE_UNIT)

ax.set_xticks(yearly_stats["Year"])
ax.tick_params(axis="x", rotation=45)

ax.legend()

save_plot(PLOTS_DIR / "05_annual_mean_median_prices.png")

# Graph 6: Annual minimum and maximum prices.
fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    yearly_stats["Year"],
    yearly_stats["Minimum_Price"],
    marker="o",
    label="Minimum price"
)

ax.plot(
    yearly_stats["Year"],
    yearly_stats["Maximum_Price"],
    marker="o",
    label="Maximum price"
)

ax.set_title("Annual Minimum and Maximum Chrysanthemum Prices")
ax.set_xlabel("Year")
ax.set_ylabel(PRICE_UNIT)

ax.set_xticks(yearly_stats["Year"])
ax.tick_params(axis="x", rotation=45)

ax.legend()

save_plot(PLOTS_DIR / "06_annual_min_max_prices.png")

# Graph 7: Yearly price volatility.
fig, ax = plt.subplots(figsize=(13, 6))

ax.bar(
    yearly_stats["Year"],
    yearly_stats["Standard_Deviation"]
)

ax.set_title("Yearly Chrysanthemum Price Volatility")
ax.set_xlabel("Year")
ax.set_ylabel("Standard Deviation of Price")

ax.set_xticks(yearly_stats["Year"])
ax.tick_params(axis="x", rotation=45)

save_plot(PLOTS_DIR / "07_yearly_price_volatility.png")

# ============================================================
# 9. MONTHLY AVERAGE PRICE FOR EACH YEAR
# ============================================================

monthly_yearly = df.groupby(
    ["Year", "Month"]
)["Price"].mean().reset_index()

monthly_yearly.rename(
    columns={"Price": "Monthly_Average_Price"},
    inplace=True
)

monthly_yearly["Month_Name"] = pd.to_datetime(
    monthly_yearly["Month"],
    format="%m"
).dt.strftime("%b")

monthly_yearly.to_csv(
    REPORTS_DIR / "monthly_average_prices_all_years.csv",
    index=False
)

month_order = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

for year in sorted(df["Year"].unique()):

    year_data = monthly_yearly[
        monthly_yearly["Year"] == year
    ].copy()

    year_data["Month_Name"] = pd.Categorical(
        year_data["Month_Name"],
        categories=month_order,
        ordered=True
    )

    year_data = year_data.sort_values("Month_Name")

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=year_data,
        x="Month_Name",
        y="Monthly_Average_Price",
        order=month_order,
        ax=ax
    )

    ax.set_title(
        f"Monthly Average Chrysanthemum Prices — {year}"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel(f"Average {PRICE_UNIT}")

    ax.tick_params(axis="x", rotation=0)

    save_plot(
        YEARLY_MONTHLY_DIR /
        f"monthly_average_price_{year}.png"
    )

# ============================================================
# 10. MONTHLY SEASONALITY ACROSS ALL YEARS
# ============================================================

monthly_seasonality = df.groupby("Month").agg(
    Mean_Price=("Price", "mean"),
    Median_Price=("Price", "median"),
    Standard_Deviation=("Price", "std"),
    Observation_Count=("Price", "count")
).reset_index()

monthly_seasonality["Month_Name"] = pd.to_datetime(
    monthly_seasonality["Month"],
    format="%m"
).dt.strftime("%b")

monthly_seasonality.to_csv(
    REPORTS_DIR / "monthly_seasonality_all_years.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(13, 6))

sns.barplot(
    data=monthly_seasonality,
    x="Month_Name",
    y="Mean_Price",
    order=month_order,
    ax=ax
)

ax.set_title(
    "Average Chrysanthemum Price by Calendar Month (2010–2025)"
)

ax.set_xlabel("Month")
ax.set_ylabel(f"Average {PRICE_UNIT}")

save_plot(PLOTS_DIR / "08_monthly_seasonality_all_years.png")

# Graph 9: Year-month heatmap.
heatmap_data = monthly_yearly.pivot(
    index="Year",
    columns="Month",
    values="Monthly_Average_Price"
)

heatmap_data = heatmap_data.reindex(
    columns=range(1, 13)
)

heatmap_data.columns = month_order

fig, ax = plt.subplots(figsize=(15, 9))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    linewidths=0.4,
    cbar_kws={"label": f"Average {PRICE_UNIT}"},
    ax=ax
)

ax.set_title("Monthly Average Chrysanthemum Prices by Year")
ax.set_xlabel("Month")
ax.set_ylabel("Year")

save_plot(PLOTS_DIR / "09_year_month_price_heatmap.png")

# ============================================================
# 11. CLASSIFY RECORDED EVENTS
# ============================================================

def contains_keyword(text, keywords):
    """Check whether text contains any configured keyword."""

    if pd.isna(text):
        return False

    text = str(text).casefold()

    return any(
        keyword.casefold() in text
        for keyword in keywords
    )


def classify_event(row):
    """
    Classify a date using explicitly configured keyword rules.

    Auspicious-day matches take priority. Unclassified event
    labels are retained as recorded events, not silently discarded.
    """

    event_text = row["Events"]

    if pd.isna(event_text):
        return "No event recorded"

    if contains_keyword(
        event_text,
        AUSPICIOUS_KEYWORDS
    ):
        return "Auspicious day"

    if contains_keyword(
        event_text,
        FESTIVAL_KEYWORDS
    ):
        return "Festival"

    return "Other recorded event"


df["Event_Category"] = df.apply(
    classify_event,
    axis=1
)

# A date can be a festival or an auspicious day only if the
# configured rules or a verified calendar identify it as such.
df["Is_Festival"] = (
    df["Event_Category"] == "Festival"
)

df["Is_Auspicious"] = (
    df["Event_Category"] == "Auspicious day"
)

df["Has_Recorded_Event"] = df["Events"].notna()

# Save event classifications for inspection and correction.
df[[
    "Date",
    "Price",
    "Tithi",
    "Masa",
    "Events",
    "Event_Category",
    "Is_Festival",
    "Is_Auspicious"
]].to_csv(
    REPORTS_DIR / "event_classification.csv",
    index=False
)

print("\n--- EVENT CLASSIFICATION ---")

print(
    df["Event_Category"].value_counts(dropna=False)
)

# ============================================================
# 12. YEARLY PRICE GRAPHS WITH FESTIVAL AND AUSPICIOUS DAYS
# ============================================================

for year in sorted(df["Year"].unique()):

    year_data = df[
        df["Year"] == year
    ].copy()

    fig, ax = plt.subplots(figsize=(16, 7))

    # Daily price series.
    ax.plot(
        year_data["Date"],
        year_data["Price"],
        linewidth=1.0,
        alpha=0.65,
        label="Daily price"
    )

    # Plot festival observations.
    festivals = year_data[
        year_data["Is_Festival"]
    ]

    if not festivals.empty:

        ax.scatter(
            festivals["Date"],
            festivals["Price"],
            marker="^",
            s=65,
            label="Festival",
            zorder=4
        )

    # Plot auspicious-day observations.
    auspicious = year_data[
        year_data["Is_Auspicious"]
    ]

    if not auspicious.empty:

        ax.scatter(
            auspicious["Date"],
            auspicious["Price"],
            marker="D",
            s=55,
            label="Auspicious day",
            zorder=5
        )

    # Plot other recorded events.
    other_events = year_data[
        year_data["Event_Category"] == "Other recorded event"
    ]

    if not other_events.empty:

        ax.scatter(
            other_events["Date"],
            other_events["Price"],
            marker="o",
            s=28,
            alpha=0.8,
            label="Other recorded event",
            zorder=3
        )

    ax.set_title(
        f"Chrysanthemum Daily Prices and Calendar Events — {year}"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(PRICE_UNIT)

    format_date_axis(ax)

    ax.legend(loc="best")

    save_plot(
        YEARLY_EVENTS_DIR /
        f"yearly_event_price_{year}.png"
    )

# ============================================================
# 13. PRICE COMPARISON BY EVENT CATEGORY
# ============================================================

event_summary = df.groupby(
    "Event_Category"
)["Price"].agg(
    Count="count",
    Mean="mean",
    Median="median",
    Minimum="min",
    Maximum="max",
    Standard_Deviation="std"
).reset_index()

event_summary.to_csv(
    REPORTS_DIR / "price_by_event_category.csv",
    index=False
)

# Boxplot by category.
fig, ax = plt.subplots(figsize=(12, 7))

sns.boxplot(
    data=df,
    x="Event_Category",
    y="Price",
    showfliers=False,
    ax=ax
)

ax.set_title(
    "Chrysanthemum Price Distribution by Calendar Event Category"
)

ax.set_xlabel("Calendar Event Category")
ax.set_ylabel(PRICE_UNIT)

ax.tick_params(axis="x", rotation=20)

save_plot(PLOTS_DIR / "10_price_by_event_category.png")

# Mean price comparison.
fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=event_summary,
    x="Event_Category",
    y="Mean",
    ax=ax
)

ax.set_title("Mean Chrysanthemum Price by Calendar Event Category")
ax.set_xlabel("Calendar Event Category")
ax.set_ylabel(f"Mean {PRICE_UNIT}")

ax.tick_params(axis="x", rotation=20)

save_plot(PLOTS_DIR / "11_mean_price_by_event_category.png")

# ============================================================
# 14. PRICE BEFORE, ON, AND AFTER EVENTS
# ============================================================

# Event windows require actual event dates.
#
# We use +/- 7 calendar days around each recorded event date.
# If multiple events are close together, windows may overlap.
# This is exploratory analysis, not a causal estimate.

event_dates = df.loc[
    df["Has_Recorded_Event"],
    ["Date", "Events", "Event_Category"]
].copy()

event_dates = event_dates.drop_duplicates(
    subset=["Date"]
)

event_windows = []

for event_row in event_dates.itertuples(index=False):

    event_date = event_row.Date
    event_name = event_row.Events
    event_category = event_row.Event_Category

    start_date = event_date - pd.Timedelta(days=7)
    end_date = event_date + pd.Timedelta(days=7)

    window = df[
        (df["Date"] >= start_date) &
        (df["Date"] <= end_date)
    ][["Date", "Price"]].copy()

    if window.empty:
        continue

    window["Event_Date"] = event_date
    window["Event_Name"] = event_name
    window["Event_Category"] = event_category

    window["Days_From_Event"] = (
        window["Date"] - event_date
    ).dt.days

    event_windows.append(window)

if event_windows:

    event_window_df = pd.concat(
        event_windows,
        ignore_index=True
    )

    event_window_df.to_csv(
        REPORTS_DIR / "event_window_daily_prices.csv",
        index=False
    )

    event_window_summary = event_window_df.groupby(
        ["Event_Category", "Days_From_Event"]
    ).agg(
        Mean_Price=("Price", "mean"),
        Median_Price=("Price", "median"),
        Observation_Count=("Price", "count")
    ).reset_index()

    event_window_summary.to_csv(
        REPORTS_DIR / "event_window_price_summary.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(13, 7))

    for category in [
        "Festival",
        "Auspicious day",
        "Other recorded event"
    ]:

        subset = event_window_summary[
            event_window_summary["Event_Category"] == category
        ]

        if subset.empty:
            continue

        ax.plot(
            subset["Days_From_Event"],
            subset["Mean_Price"],
            marker="o",
            label=category
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
        label="Event date"
    )

    ax.set_title(
        "Average Chrysanthemum Prices Before and After Recorded Events"
    )

    ax.set_xlabel("Days Before / After Event")
    ax.set_ylabel(f"Mean {PRICE_UNIT}")

    ax.set_xticks(range(-7, 8))

    ax.legend()

    save_plot(
        PLOTS_DIR / "12_event_window_price_analysis.png"
    )

# ============================================================
# 15. DAY-OF-WEEK PRICE ANALYSIS
# ============================================================

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_summary = df.groupby(
    "Day_Of_Week"
)["Price"].agg(
    Mean="mean",
    Median="median",
    Count="count"
).reindex(weekday_order).reset_index()

weekday_summary.to_csv(
    REPORTS_DIR / "weekday_price_statistics.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=weekday_summary,
    x="Day_Of_Week",
    y="Mean",
    order=weekday_order,
    ax=ax
)

ax.set_title("Average Chrysanthemum Price by Day of the Week")
ax.set_xlabel("Day of the Week")
ax.set_ylabel(f"Mean {PRICE_UNIT}")

ax.tick_params(axis="x", rotation=20)

save_plot(PLOTS_DIR / "13_weekday_price_analysis.png")

# ============================================================
# 16. PRICE CHANGE ANALYSIS
# ============================================================

df["Daily_Price_Change"] = df["Price"].diff()

df["Daily_Price_Change_Percent"] = (
    df["Price"].pct_change() * 100
)

# The percentage change is undefined when the previous price is zero.
df.loc[
    df["Price"].shift(1) == 0,
    "Daily_Price_Change_Percent"
] = np.nan

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    df["Date"],
    df["Daily_Price_Change"],
    linewidth=0.8,
    alpha=0.7
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_title("Daily Change in Chrysanthemum Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Change in Price")

format_date_axis(ax)

save_plot(PLOTS_DIR / "14_daily_price_change.png")

# ============================================================
# 17. YEAR-OVER-YEAR PRICE CHANGES
# ============================================================

fig, ax = plt.subplots(figsize=(13, 6))

ax.bar(
    yearly_stats["Year"],
    yearly_stats["YoY_Mean_Change_Percent"]
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_title("Year-over-Year Change in Annual Mean Price")
ax.set_xlabel("Year")
ax.set_ylabel("Change in Mean Price (%)")

ax.set_xticks(yearly_stats["Year"])
ax.tick_params(axis="x", rotation=45)

save_plot(PLOTS_DIR / "15_year_over_year_price_change.png")

# ============================================================
# 18. MONTHLY PRICE DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

sns.boxplot(
    data=df,
    x="Month_Name",
    y="Price",
    order=month_order,
    showfliers=False,
    ax=ax
)

ax.set_title("Chrysanthemum Price Distribution by Month")
ax.set_xlabel("Month")
ax.set_ylabel(PRICE_UNIT)

save_plot(PLOTS_DIR / "16_monthly_price_distribution.png")

# ============================================================
# 19. YEARLY OUTLIER SUMMARY
# ============================================================

yearly_outliers = df.groupby("Year").agg(
    Total_Observations=("Price", "count"),
    Outlier_Count=("Is_Outlier", "sum")
).reset_index()

yearly_outliers["Outlier_Percentage"] = (
    yearly_outliers["Outlier_Count"] /
    yearly_outliers["Total_Observations"] * 100
)

yearly_outliers.to_csv(
    REPORTS_DIR / "yearly_outlier_summary.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(13, 6))

ax.bar(
    yearly_outliers["Year"],
    yearly_outliers["Outlier_Count"]
)

ax.set_title("Number of IQR-Detected Price Outliers per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Outliers")

ax.set_xticks(yearly_outliers["Year"])
ax.tick_params(axis="x", rotation=45)

save_plot(PLOTS_DIR / "17_yearly_outlier_counts.png")

# ============================================================
# 20. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("EDA COMPLETED")
print("=" * 65)

print("\nPlots saved to:")
print(PLOTS_DIR.resolve())

print("\nStatistical reports saved to:")
print(REPORTS_DIR.resolve())

print("\nMonthly average price graphs saved to:")
print(YEARLY_MONTHLY_DIR.resolve())

print("\nYearly event graphs saved to:")
print(YEARLY_EVENTS_DIR.resolve())

print("\nTotal plots:", len(list(PLOTS_DIR.rglob("*.png"))))
print("Total report CSV files:", len(list(REPORTS_DIR.glob("*.csv"))))

print("\nImportant:")
print("- IQR outliers are statistical flags, not necessarily errors.")
print("- Event classifications depend on the configured keywords.")
print("- No missing price values were imputed.")
print("- Check market units before interpreting price magnitudes.")

print("\nEDA finished successfully.")