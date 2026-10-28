
import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# 1. INPUT FILE PATH (UNCHANGED)
# --------------------------------------------------

input_file = (
    r"C:\Users\Ghanavi Arun\Floriculture\data\silver"
    r"\chrysanthemum_adjusted.csv"
)

# --------------------------------------------------
# 2. EDA FOLDER PATH (UNCHANGED)
# --------------------------------------------------

eda_folder = (
    r"C:\Users\Ghanavi Arun\Floriculture\code\eda"
)

# Check paths
if not os.path.isdir(eda_folder):
    raise FileNotFoundError(
        f"EDA folder does not exist: {eda_folder}"
    )

if not os.path.isfile(input_file):
    raise FileNotFoundError(
        f"Input file not found: {input_file}"
    )

# --------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# --------------------------------------------------

df = pd.read_csv(input_file)

required_columns = ["Date", "Price"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
).dt.normalize()

df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)

df = df.dropna(
    subset=["Date", "Price"]
).copy()

df = df.sort_values("Date").reset_index(drop=True)

if df["Date"].duplicated().any():
    duplicate_dates = (
        df.loc[df["Date"].duplicated(False), "Date"]
        .dt.strftime("%Y-%m-%d")
        .unique()
    )

    raise ValueError(
        "Duplicate dates found. Resolve them before "
        f"analysis. Examples: {duplicate_dates[:10]}"
    )

if (df["Price"] <= 0).any():
    raise ValueError(
        "Non-positive prices found. Review them first."
    )

if df.empty:
    raise ValueError("No valid observations found.")

# --------------------------------------------------
# 4. CHECK CALENDAR-DATE CONTINUITY
# --------------------------------------------------

# Check whether every calendar day is present.
expected_dates = pd.date_range(
    start=df["Date"].min(),
    end=df["Date"].max(),
    freq="D"
)

missing_dates = expected_dates.difference(df["Date"])

print("\nCALENDAR CONTINUITY CHECK")
print("=" * 50)
print(f"Valid observations: {len(df)}")
print(f"Missing calendar dates: {len(missing_dates)}")

if len(missing_dates) > 0:
    print("First 10 missing dates:")
    print(missing_dates[:10].strftime("%Y-%m-%d").tolist())

# --------------------------------------------------
# 5. IDENTIFY HIGH-PRICE DAYS
# --------------------------------------------------

# Dataset-wide 95th percentile.
# This identifies unusually high prices, not necessarily
# sudden price spikes.

high_price_threshold = df["Price"].quantile(0.95)

df["High_Price_Threshold"] = high_price_threshold

df["Is_High_Price_Day"] = (
    df["Price"] >= high_price_threshold
).astype(int)

# --------------------------------------------------
# 6. CALCULATE PREVIOUS 7 CALENDAR-DAY STATISTICS
# --------------------------------------------------

# Reindex to calendar days so missing dates cannot cause
# the rolling window to silently span extra calendar days.
#
# Missing prices remain NaN; they are NOT interpolated.

daily = (
    df.set_index("Date")[["Price"]]
    .reindex(expected_dates)
)

daily.index.name = "Date"

# Shift first: exclude the current day's price.
previous_prices = daily["Price"].shift(1)

daily["Previous_7_Day_Mean"] = (
    previous_prices
    .rolling(window=7, min_periods=7)
    .mean()
)

daily["Previous_7_Day_Std"] = (
    previous_prices
    .rolling(window=7, min_periods=7)
    .std()
)

daily["Local_Spike_Threshold"] = (
    daily["Previous_7_Day_Mean"]
    + 2 * daily["Previous_7_Day_Std"]
)

# --------------------------------------------------
# 7. IDENTIFY LOCAL PRICE SPIKES
# --------------------------------------------------

# Require seven consecutive valid previous daily prices.
# If any previous calendar day is missing, no local-spike
# decision is made for the current day.

daily["Is_Local_Price_Spike"] = pd.Series(
    pd.NA,
    index=daily.index,
    dtype="Int64"
)

valid_history = (
    previous_prices
    .rolling(window=7, min_periods=7)
    .count()
    .eq(7)
    & daily["Price"].notna()
)

daily.loc[valid_history, "Is_Local_Price_Spike"] = (
    daily.loc[valid_history, "Price"]
    > daily.loc[valid_history, "Local_Spike_Threshold"]
).astype(int)

# --------------------------------------------------
# 8. CALCULATE DAILY PRICE CHANGES
# --------------------------------------------------

daily["Previous_Day_Price"] = daily["Price"].shift(1)

daily["Daily_Price_Change"] = (
    daily["Price"] - daily["Previous_Day_Price"]
)

daily["Daily_Price_Change_Percent"] = (
    daily["Price"].pct_change(fill_method=None) * 100
)

# --------------------------------------------------
# 9. MERGE RESULTS BACK INTO THE FULL DAILY DATASET
# --------------------------------------------------

result_columns = [
    "Previous_7_Day_Mean",
    "Previous_7_Day_Std",
    "Local_Spike_Threshold",
    "Is_Local_Price_Spike",
    "Previous_Day_Price",
    "Daily_Price_Change",
    "Daily_Price_Change_Percent"
]

df = df.merge(
    daily[result_columns].reset_index(),
    on="Date",
    how="left",
    validate="one_to_one"
)

# --------------------------------------------------
# 10. CREATE A COMBINED FLAG FOR REPORTING ONLY
# --------------------------------------------------

# 1 = high-price day OR local price spike.
# Keep the original two flags separate for analysis.

df["Is_High_or_Local_Spike"] = (
    (df["Is_High_Price_Day"] == 1)
    | (df["Is_Local_Price_Spike"] == 1)
).astype(int)

# --------------------------------------------------
# 11. SAVE COMPLETE DAILY DATASET
# --------------------------------------------------

full_output = os.path.join(
    eda_folder,
    "chrysanthemum_daily_spike_analysis.csv"
)

df.to_csv(full_output, index=False)

# --------------------------------------------------
# 12. SAVE ALL FLAGGED DATES
# --------------------------------------------------

spikes = df[
    df["Is_High_or_Local_Spike"] == 1
].copy()

spike_output = os.path.join(
    eda_folder,
    "chrysanthemum_all_price_spikes.csv"
)

spikes.to_csv(spike_output, index=False)

# --------------------------------------------------
# 13. SAVE SEPARATE SPIKE SUMMARY
# --------------------------------------------------

summary_columns = [
    "Date",
    "Price",
    "Is_High_Price_Day",
    "Is_Local_Price_Spike",
    "Is_High_or_Local_Spike",
    "Previous_7_Day_Mean",
    "Previous_7_Day_Std",
    "Local_Spike_Threshold",
    "Daily_Price_Change",
    "Daily_Price_Change_Percent"
]

summary_file = os.path.join(
    eda_folder,
    "chrysanthemum_spike_summary.csv"
)

spikes[summary_columns].to_csv(
    summary_file,
    index=False
)

# --------------------------------------------------
# 14. DISPLAY SUMMARY
# --------------------------------------------------

print("\nPRICE SPIKE ANALYSIS COMPLETED")
print("=" * 50)

print(f"Total valid observations: {len(df)}")

print(
    f"Date range: {df['Date'].min().date()} "
    f"to {df['Date'].max().date()}"
)

print(
    f"\n95th percentile threshold: "
    f"{high_price_threshold:.2f}"
)

print(
    "Total high-price days: "
    f"{df['Is_High_Price_Day'].sum()}"
)

print(
    "Total local price spikes: "
    f"{df['Is_Local_Price_Spike'].sum(skipna=True)}"
)

print(
    "Days without enough valid history: "
    f"{df['Is_Local_Price_Spike'].isna().sum()}"
)

print(
    "Total unique dates flagged by either method: "
    f"{len(spikes)}"
)

print(f"\nFull daily dataset:\n{full_output}")
print(f"\nAll flagged dates:\n{spike_output}")
print(f"\nSpike summary:\n{summary_file}")

print("\nFirst 10 flagged dates:")

if spikes.empty:
    print("No high-price days or local spikes identified.")
else:
    print(
        spikes[summary_columns]
        .head(10)
        .to_string(index=False)
    )

print("\nAnalysis finished successfully.")