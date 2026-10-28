import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================
# 0. Configuration & Output Paths
# ============================================================
EVENTS_PATH = Path(r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\hindu_calendar_1900_2100.csv")
PRED_PATH   = Path(r"C:\Users\Ghanavi Arun\Floriculture\code\ml_training\model5\model5a_test_predictions.csv")

# Exact output directory requested
OUT_DIR = Path(r"C:\Users\Ghanavi Arun\Floriculture\code\ml_training\model5\plot_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "festival_price_spikes.png"
OUT_CSV = OUT_DIR / "Festival_Real_vs_Predicted_Prices.csv"

# ============================================================
# 1. Helpers
# ============================================================
def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df

def find_column(df, candidates):
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None

def load_test_data_for_actuals(pred_df, pred_path):
    """Tries to find the original test dataset to grab the REAL prices."""
    test_files = [
        pred_path.parent / "model5a_test.csv",
        pred_path.parent / "model5a_test_data.csv",
        pred_path.parent / "test.csv",
        pred_path.parent / "test_data.csv",
    ]
    for f in test_files:
        if f.exists() and f.resolve() != pred_path.resolve():
            try:
                test_df = normalize_columns(pd.read_csv(f))
                # Look for actual price column
                actual_col = find_column(test_df, ["price", "actual_price", "original_price", "target", "y_true"])
                date_col = find_column(test_df, ["date", "ds", "timestamp", "datetime"])
                
                if actual_col and len(test_df) == len(pred_df):
                    print(f"[info] Found actual prices in {f.name}")
                    if date_col:
                        pred_df["date"] = pd.to_datetime(test_df[date_col], errors="coerce")
                    pred_df["original_price"] = pd.to_numeric(test_df[actual_col], errors="coerce")
                    return pred_df
            except Exception as e:
                continue
    return pred_df

def load_predictions(path):
    raw = normalize_columns(pd.read_csv(path))
    
    # 1. Find Predicted Price
    pred_col = find_column(raw, ["predicted_price", "predicted", "prediction", "y_pred", "pred", "forecast"])
    if pred_col is None:
        raise ValueError(f"No prediction column found in {path.name}. Columns: {raw.columns.tolist()}")
    df = raw.rename(columns={pred_col: "predicted_price"})
    df["predicted_price"] = pd.to_numeric(df["predicted_price"], errors="coerce")

    # 2. Find or Recover Actual Price
    actual_col = find_column(raw, ["original_price", "actual_price", "y_true", "actual", "true_price", "price"])
    if actual_col is not None:
        df = df.rename(columns={actual_col: "original_price"})
        df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
    else:
        # Try to find actuals in companion test file
        df = load_test_data_for_actuals(df, path)

    # 3. Find Date
    date_col = find_column(raw, ["date", "ds", "timestamp", "datetime"])
    if date_col is not None:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif "date" not in df.columns:
        # If date is a saved index (Unnamed: 0)
        for col in raw.columns:
            if str(col).lower().startswith("unnamed"):
                try:
                    d = pd.to_datetime(raw[col], errors="coerce")
                    if d.notna().mean() > 0.9:
                        df["date"] = d
                        break
                except:
                    pass
                    
    if "date" not in df.columns:
        raise ValueError(f"Could not find or recover dates for {path.name}.")

    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df

def load_events(path):
    df = normalize_columns(pd.read_csv(path))
    date_col = find_column(df, ["date"])
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)

# ============================================================
# 2. Load Data & Merge
# ============================================================
events = load_events(EVENTS_PATH)
prices = load_predictions(PRED_PATH)

events = events[(events["date"] >= prices["date"].min()) & (events["date"] <= prices["date"].max())].copy()

events_per_day = (
    events.groupby("date")["title"]
    .apply(lambda s: "; ".join(sorted({str(x) for x in s if pd.notna(x)})))
    .reset_index()
    .rename(columns={"title": "festival_title"})
)

MAJOR = {
    "Diwali", "Holi", "Dussehra", "Vijayadashami", "Maha Shivaratri",
    "Ganesh Chaturthi", "Navratri Begins", "Raksha Bandhan", "Rakhi",
    "Janmashtami *Smarta", "Krishna Janmashtami", "Rama Navami",
    "Hanuman Jayanti", "Akshaya Tritiya", "Dhanteras", "Lakshmi Puja",
    "Makar Sankranti", "Makara Sankranti", "Pongal", "Onam",
    "Chhath Puja", "Karwa Chauth", "Teej", "Hartalika Teej",
    "Gudi Padwa", "Ugadi", "Basant Panchami", "Vasant Panchami",
    "Guru Purnima", "Nag Panchami", "Rath Yatra", "Jagannath Rathyatra",
}

events_per_day["is_major"] = events_per_day["festival_title"].apply(lambda x: any(m in x for m in MAJOR))

df = prices.merge(events_per_day, on="date", how="left").sort_values("date").reset_index(drop=True)
df["is_event"] = df["festival_title"].notna()
df["is_major_event"] = df["is_major"].fillna(False).astype(bool)

# ============================================================
# 3. Spike detection
# ============================================================
df = df.set_index("date")
df["baseline"] = df["predicted_price"].rolling(30, center=True, min_periods=5).median()
df["pct_vs_baseline"] = (df["predicted_price"] / df["baseline"] - 1) * 100
df = df.reset_index()

# ============================================================
# 4. Plot
# ============================================================
fig, ax = plt.subplots(figsize=(18, 7))
ax.plot(df["date"], df["predicted_price"], color="#1f77b4", lw=1.0, label="Predicted price")
ax.plot(df["date"], df["baseline"], color="grey", lw=1.0, ls="--", label="30-day rolling median")
major = df.loc[df["is_major_event"]]
ax.scatter(major["date"], major["predicted_price"], s=40, color="crimson", zorder=5, label="Major festival")

ax.set_title("Predicted price vs Hindu festival events", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Predicted price", fontsize=12)

span_days = (df["date"].max() - df["date"].min()).days
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if span_days <= 1200 else 6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45, ha='right')
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"[info] Saved graph → {OUT_FIG}")
plt.show()

# ============================================================
# 5. EXPORT THE CSV WITH REAL & PREDICTED PRICES
# ============================================================
print("\n" + "="*60)
print("GENERATING FESTIVAL PRICES CSV")
print("="*60)

# Filter only days with festivals
festival_days = df[df["is_event"]].copy()

# Build the final output DataFrame
out_cols = {
    "date": "Date",
    "festival_title": "Festival_Event",
    "original_price": "Real_Price",
    "predicted_price": "Predicted_Price"
}

# Check if Real_Price exists
if "original_price" not in festival_days.columns:
    print("[WARNING] Could not find 'Real Price' (actuals) in the files.")
    print("The CSV will only contain Predicted Prices.")
    del out_cols["original_price"]

# Extract and rename
final_csv = festival_days[list(out_cols.keys())].rename(columns=out_cols)
final_csv = final_csv.sort_values("Date")

# Print to console
print(final_csv.to_string(index=False))

# Save to disk
final_csv.to_csv(OUT_CSV, index=False)
print(f"\n[SUCCESS] CSV saved to exactly: {OUT_CSV}")