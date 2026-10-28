
import pandas as pd
import os

# --------------------------------------------------
# 1. INPUT AND OUTPUT PATHS
# --------------------------------------------------

input_file = (
    r"C:\Users\Ghanavi Arun\Floriculture"
    r"\code\feature_engineering\baseline_features.csv"
)

output_file = (
    r"C:\Users\Ghanavi Arun\Floriculture"
    r"\code\feature_engineering"
    r"\festival_auspicious_window_features.csv"
)


# --------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# --------------------------------------------------

df = pd.read_csv(input_file)

original_rows = len(df)
original_columns = df.columns.tolist()

print("Original dataset loaded successfully.")
print("Original rows:", original_rows)
print("Original columns:", len(original_columns))


# --------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "Date",
    "is_festival",
    "is_auspicious"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from input: {missing_columns}"
    )


# --------------------------------------------------
# 4. PREPARE DATES WITHOUT REMOVING ROWS
# --------------------------------------------------

# Use a separate date series for calculations.
# The original Date column and row order are preserved.
dates = pd.to_datetime(
    df["Date"],
    errors="coerce"
).dt.normalize()


# --------------------------------------------------
# 5. PREPARE FESTIVAL AND AUSPICIOUS FLAGS
# --------------------------------------------------

def prepare_flag(series, column_name):

    text = series.astype("string").str.strip().str.lower()

    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
        "": 0,
        "nan": 0,
        "none": 0
    }

    result = pd.to_numeric(series, errors="coerce")
    result = result.fillna(text.map(mapping))

    # Do not silently treat unexpected nonblank values as zero.
    unknown = result.isna() | ~result.isin([0, 1])

    if unknown.any():
        bad_values = series[unknown].unique()[:10]

        raise ValueError(
            f"Unexpected values in {column_name}: {bad_values}"
        )

    return result.astype(int)


festival_flag = prepare_flag(
    df["is_festival"],
    "is_festival"
)

auspicious_flag = prepare_flag(
    df["is_auspicious"],
    "is_auspicious"
)


# --------------------------------------------------
# 6. IDENTIFY EVENT DATES
# --------------------------------------------------

festival_dates = set(
    dates[festival_flag.eq(1) & dates.notna()]
)

auspicious_dates = set(
    dates[auspicious_flag.eq(1) & dates.notna()]
)

print("\nUnique festival dates:", len(festival_dates))
print("Unique auspicious dates:", len(auspicious_dates))


# --------------------------------------------------
# 7. CREATE ±3-DAY FESTIVAL MARKERS
# --------------------------------------------------

# BEFORE means the target date occurs 1, 2, or 3 days
# before a festival.
#
# AFTER means the target date occurs 1, 2, or 3 days
# after a festival.
#
# All calculations use actual calendar dates.

for days in [1, 2, 3]:

    df[f"Festival_{days}_Day_Before"] = dates.apply(
        lambda d: int(
            pd.notna(d) and
            (d + pd.Timedelta(days=days)) in festival_dates
        )
    )

    df[f"Festival_{days}_Day_After"] = dates.apply(
        lambda d: int(
            pd.notna(d) and
            (d - pd.Timedelta(days=days)) in festival_dates
        )
    )


# --------------------------------------------------
# 8. CREATE ±3-DAY AUSPICIOUS-DAY MARKERS
# --------------------------------------------------

for days in [1, 2, 3]:

    df[f"Auspicious_{days}_Day_Before"] = dates.apply(
        lambda d: int(
            pd.notna(d) and
            (d + pd.Timedelta(days=days)) in auspicious_dates
        )
    )

    df[f"Auspicious_{days}_Day_After"] = dates.apply(
        lambda d: int(
            pd.notna(d) and
            (d - pd.Timedelta(days=days)) in auspicious_dates
        )
    )


# --------------------------------------------------
# 9. CREATE COMBINED EVENT-WINDOW MARKERS
# --------------------------------------------------

# Includes the exact event date and dates up to 3 days
# before or after the event.

def within_event_window(date, event_dates):

    if pd.isna(date):
        return 0

    for offset in range(-3, 4):

        if date + pd.Timedelta(days=offset) in event_dates:
            return 1

    return 0


df["Festival_Within_3_Days"] = dates.apply(
    lambda d: within_event_window(d, festival_dates)
)

df["Auspicious_Within_3_Days"] = dates.apply(
    lambda d: within_event_window(d, auspicious_dates)
)

all_event_dates = festival_dates | auspicious_dates

df["Any_Event_Within_3_Days"] = dates.apply(
    lambda d: within_event_window(d, all_event_dates)
)


# --------------------------------------------------
# 10. RETAIN ALL ORIGINAL ROWS AND COLUMNS
# --------------------------------------------------

# No rows are dropped.
# Every original column is preserved.
# Only new marker columns are appended.

if len(df) != original_rows:
    raise RuntimeError(
        "Row count changed unexpectedly. Output was not saved."
    )

if df[original_columns].shape != (
    original_rows,
    len(original_columns)
):
    raise RuntimeError(
        "Original columns were not preserved correctly."
    )


# --------------------------------------------------
# 11. SAVE NEW DATASET
# --------------------------------------------------

output_folder = os.path.dirname(output_file)

if not os.path.isdir(output_folder):
    raise FileNotFoundError(
        f"Output folder does not exist: {output_folder}"
    )

df.to_csv(output_file, index=False)


# --------------------------------------------------
# 12. DISPLAY SUMMARY
# --------------------------------------------------

new_columns = [
    col for col in df.columns
    if col not in original_columns
]

print("\nFeature engineering completed successfully.")

print("\nInput file:")
print(input_file)

print("\nOutput file:")
print(output_file)

print("\nOriginal row count:", original_rows)
print("Final row count:", len(df))

print("Original column count:", len(original_columns))
print("New columns added:", len(new_columns))
print("Final column count:", len(df.columns))

print("\nNew columns:")
for column in new_columns:
    print(column)

print("\nMarker counts:")
for column in new_columns:
    print(f"{column}: {df[column].sum()}")

print("\nFirst 5 rows of the new dataset:")
print(df.head())

print("\nAll original rows have been retained.")
print("The original input file has not been modified.")
