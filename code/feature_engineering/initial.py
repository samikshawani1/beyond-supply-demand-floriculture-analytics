
import pandas as pd
import os

# --------------------------------------------------
# 1. Define input and output file paths
# --------------------------------------------------

input_file = r"C:\Users\Ghanavi Arun\Floriculture\data\silver\chrysanthemum_adjusted.csv"

output_file = r"C:\Users\Ghanavi Arun\Floriculture\code\feature_engineering\baseline_features.csv"


# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

original_row_count = len(df)

print("Original dataset loaded successfully.")
print("Original number of rows:", original_row_count)
print("Original columns:", df.columns.tolist())


# --------------------------------------------------
# 3. Validate required columns
# --------------------------------------------------

required_columns = ["Date", "Price"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found "
            f"in the input CSV."
        )


# --------------------------------------------------
# 4. Prepare Date and Price columns
# --------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Sort records chronologically.
# Do not remove rows with missing dates or prices.
df = df.sort_values("Date", na_position="last").reset_index(drop=True)

# Check for duplicate valid dates
if df["Date"].dropna().duplicated().any():
    duplicate_dates = df.loc[
        df["Date"].notna() &
        df["Date"].duplicated(keep=False),
        "Date"
    ].unique()

    raise ValueError(
        "Duplicate dates found. Resolve them before feature "
        f"engineering. Examples: {duplicate_dates[:10]}"
    )


# --------------------------------------------------
# 5. Create historical price lag features
# --------------------------------------------------

lag_periods = [1, 2, 3, 7, 14]

for lag in lag_periods:
    df[f"price_lag_{lag}"] = df["Price"].shift(lag)


# --------------------------------------------------
# 6. Create rolling mean features
# --------------------------------------------------

# Exclude the current row's price from historical averages
previous_prices = df["Price"].shift(1)

rolling_windows = [3, 7, 14]

for window in rolling_windows:
    df[f"rolling_mean_{window}"] = (
        previous_prices
        .rolling(window=window, min_periods=window)
        .mean()
    )


# --------------------------------------------------
# 7. Create rolling standard deviation
# --------------------------------------------------

df["rolling_std_7"] = (
    previous_prices
    .rolling(window=7, min_periods=7)
    .std()
)


# --------------------------------------------------
# 8. Create price change features
# --------------------------------------------------

df["price_change_1d"] = (
    df["Price"].shift(1) - df["Price"].shift(2)
)

df["price_pct_change_1d"] = (
    df["Price"].shift(1).pct_change(fill_method=None) * 100
)


# --------------------------------------------------
# 9. Add calendar features
# --------------------------------------------------

df["Day_Of_Week"] = df["Date"].dt.dayofweek
# Monday = 0, Sunday = 6

df["Day_Of_Month"] = df["Date"].dt.day

df["Month"] = df["Date"].dt.month

df["Year"] = df["Date"].dt.year

df["Day_Of_Year"] = df["Date"].dt.dayofyear


# --------------------------------------------------
# 10. Retain ALL original rows
# --------------------------------------------------

# Do not use dropna().
# Missing lag and rolling values are expected for early rows.
df_features = df.copy()


# --------------------------------------------------
# 11. Save the feature-engineered dataset
# --------------------------------------------------

output_folder = os.path.dirname(output_file)

if output_folder:
    os.makedirs(output_folder, exist_ok=True)

df_features.to_csv(output_file, index=False)


# --------------------------------------------------
# 12. Display completion details
# --------------------------------------------------

print("\nFeature engineering completed successfully.")

print("Input file:", input_file)
print("Output file:", output_file)

print("Original number of rows:", original_row_count)
print("Final number of rows:", len(df_features))
print("Rows retained:", len(df_features))
print("Rows removed:", original_row_count - len(df_features))

print("\nNew features created:")

print([
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_7",
    "price_lag_14",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "price_change_1d",
    "price_pct_change_1d",
    "Day_Of_Week",
    "Day_Of_Month",
    "Month",
    "Year",
    "Day_Of_Year"
])

print("\nFinal dataset shape:", df_features.shape)

print("\nFinal dataset columns:")
print(df_features.columns.tolist())

print("\nMissing values in new features:")

feature_columns = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_7",
    "price_lag_14",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "price_change_1d",
    "price_pct_change_1d",
    "Day_Of_Week",
    "Day_Of_Month",
    "Month",
    "Year",
    "Day_Of_Year"
]

print(df_features[feature_columns].isna().sum())

print("\nFirst 5 rows:")
print(df_features.head())

print("\nAll rows have been retained and saved.")