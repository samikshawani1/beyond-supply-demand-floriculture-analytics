import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

INPUT_FILE = Path(
    r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\all_data_classified.csv"
)

# CSV and log output folder
OUTPUT_DIR = Path(
    r"C:\Users\Ghanavi Arun\Floriculture\data\silver"
)

OUTPUT_FILE = OUTPUT_DIR / "chrysanthemum_adjusted.csv"
LOG_FILE = OUTPUT_DIR / "price_adjustment_log.csv"

# Graph image output folder
GRAPH_DIR = Path(
    r"C:\Users\Ghanavi Arun\Floriculture\code\preprocessing"
)

GRAPH_FILE = GRAPH_DIR / "price_adjustment_graph.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input CSV not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# Check required columns
if "Date" not in df.columns or "Price" not in df.columns:
    raise ValueError("Dataset must contain Date and Price columns.")

df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

if df["Price"].isna().any():
    raise ValueError(
        "Price contains missing or non-numeric values. "
        "Investigate them before adjustment."
    )

# Preserve the original row order
df["_original_order"] = np.arange(len(df))

# Sort chronologically for neighboring-price calculations
df = df.sort_values("Date").reset_index(drop=True)

# --------------------------------------------------
# 3. DEFINE PRICE LIMITS
# --------------------------------------------------

MIN_PRICE = 50
MAX_PRICE = 2100

# Keep original prices for calculations
original_prices = df["Price"].copy()

# Identify prices outside the allowed range
outlier_mask = (
    (original_prices < MIN_PRICE) |
    (original_prices > MAX_PRICE)
)

outlier_positions = np.flatnonzero(outlier_mask.to_numpy())

# Store adjusted prices separately
adjusted_prices = original_prices.copy()

adjustment_log = []

# --------------------------------------------------
# 4. REPLACE OUTLIER PRICES
# --------------------------------------------------

for i in outlier_positions:

    # Previous two and next two observations
    neighbor_positions = [
        j for j in [i - 2, i - 1, i + 1, i + 2]
        if 0 <= j < len(df)
    ]

    # Use only valid ORIGINAL neighboring prices
    valid_positions = [
        j for j in neighbor_positions
        if MIN_PRICE <= original_prices.iloc[j] <= MAX_PRICE
    ]

    method = (
        "Mean of valid prices within previous 2 and next 2 observations"
    )

    # Fallback if all four neighboring prices are invalid
    if not valid_positions:

        distance = 3

        while distance < len(df) and not valid_positions:

            candidates = [
                j for j in [i - distance, i + distance]
                if 0 <= j < len(df)
            ]

            valid_positions = [
                j for j in candidates
                if MIN_PRICE <= original_prices.iloc[j] <= MAX_PRICE
            ]

            distance += 1

        method = "Fallback: nearest valid original price observations"

    if not valid_positions:
        raise ValueError(
            f"No valid neighboring prices found at sorted row {i}."
        )

    # Calculate replacement using ORIGINAL neighboring prices
    replacement_price = original_prices.iloc[valid_positions].mean()

    # Replace only in the adjusted copy
    adjusted_prices.iloc[i] = replacement_price

    # Record the adjustment
    adjustment_log.append({
        "Date": df.loc[i, "Date"],
        "Original_Price": original_prices.iloc[i],
        "Adjusted_Price": replacement_price,
        "Neighbor_Dates": ", ".join(
            df.loc[valid_positions, "Date"].dt.strftime("%Y-%m-%d")
        ),
        "Neighbor_Prices": ", ".join(
            str(original_prices.iloc[j]) for j in valid_positions
        ),
        "Method": method
    })

# --------------------------------------------------
# 5. SAVE ADJUSTED DATASET
# --------------------------------------------------

df["Price"] = adjusted_prices

# Confirm all prices are within the required limits
if not df["Price"].between(MIN_PRICE, MAX_PRICE).all():
    raise ValueError(
        "Some prices remain outside the required range of 50 to 2100."
    )

# Restore original row order
df = df.sort_values("_original_order")

df.drop(columns=["_original_order"], inplace=True)

# Restore date formatting
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

# Save a NEW CSV; original Bronze CSV remains unchanged
df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 6. SAVE ADJUSTMENT LOG
# --------------------------------------------------

log_columns = [
    "Date",
    "Original_Price",
    "Adjusted_Price",
    "Neighbor_Dates",
    "Neighbor_Prices",
    "Method"
]

log_df = pd.DataFrame(adjustment_log, columns=log_columns)
log_df.to_csv(LOG_FILE, index=False)

# --------------------------------------------------
# 7. SAVE GRAPH IMAGE
# --------------------------------------------------

# Read original data for the graph
original_df_for_graph = pd.read_csv(INPUT_FILE)

original_graph_data = pd.DataFrame({
    "Date": pd.to_datetime(
        original_df_for_graph["Date"], errors="coerce"
    ),
    "Original_Price": pd.to_numeric(
        original_df_for_graph["Price"], errors="coerce"
    )
}).dropna(
    subset=["Date", "Original_Price"]
).sort_values("Date")

# Prepare adjusted data for the graph
graph_data = pd.DataFrame({
    "Date": pd.to_datetime(df["Date"]),
    "Adjusted_Price": pd.to_numeric(df["Price"])
}).sort_values("Date")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    original_graph_data["Date"],
    original_graph_data["Original_Price"],
    label="Original price",
    linewidth=1,
    alpha=0.7
)

ax.plot(
    graph_data["Date"],
    graph_data["Adjusted_Price"],
    label="Adjusted price",
    linewidth=1.2,
    alpha=0.9
)

ax.axhline(
    MIN_PRICE,
    linestyle="--",
    linewidth=1,
    label=f"Minimum allowed price: {MIN_PRICE}"
)

ax.axhline(
    MAX_PRICE,
    linestyle="--",
    linewidth=1,
    label=f"Maximum allowed price: {MAX_PRICE}"
)

ax.set_title("Chrysanthemum Prices Before and After Adjustment")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.grid(alpha=0.25)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(GRAPH_FILE, dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# 8. DISPLAY SUMMARY
# --------------------------------------------------

print("\nPRICE ADJUSTMENT COMPLETED")
print("-" * 40)

print("Original file:", INPUT_FILE)
print("Adjusted file:", OUTPUT_FILE)
print("Adjustment log:", LOG_FILE)
print("Graph image:", GRAPH_FILE)

print("Total records:", len(df))
print("Prices adjusted:", len(adjustment_log))

print("Minimum adjusted price:", pd.to_numeric(df["Price"]).min())
print("Maximum adjusted price:", pd.to_numeric(df["Price"]).max())

print("\nOriginal dataset remains unchanged.")