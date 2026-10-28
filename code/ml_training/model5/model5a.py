import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================
# 0. Configuration & Exact Paths
# ============================================================
FEATURES_PATH = Path(r"C:\Users\Ghanavi Arun\Floriculture\code\feature_engineering\festival_auspicious_window_features.csv")
PRED_PATH     = Path(r"C:\Users\Ghanavi Arun\Floriculture\code\ml_training\model5\model5a_test_predictions.csv")
EVENTS_PATH   = Path(r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\hindu_calendar_1900_2100.csv")

OUT_DIR = Path(r"C:\Users\Ghanavi Arun\Floriculture\code\ml_training\model5\plot_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "festival_price_spikes.png"
OUT_CSV = OUT_DIR / "Festival_Real_vs_Predicted_Prices.csv"

# ============================================================
# 1. Helpers
# ============================================================
def find_column(df, candidates):
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None

# ============================================================
# 2. Load and Align Data
# ============================================================
print("[info] Loading data...")

# --- A. Load Features (Real Prices & Dates) ---
features = pd.read_csv(FEATURES_PATH)
features.columns = [str(c).strip().lstrip("\ufeff") for c in features.columns]

date_col = find_column(features, ["date", "ds", "timestamp", "datetime"])
real_col = find_column(features, ["price", "modal_price", "actual_price", "target", "y", "original_price"])

if not date_col or not real_col:
    raise ValueError(f"Could not find Date or Real Price in features file. Columns: {features.columns.tolist()}")

features = features.rename(columns={date_col: "date", real_col: "original_price"})
features["date"] = pd.to_datetime(features["date"])
features = features[["date", "original_price"]].sort_values("date").reset_index(drop=True)

# --- B. Load Predictions ---
preds = pd.read_csv(PRED_PATH)
preds.columns = [str(c).strip().lstrip("\ufeff") for c in preds.columns]

pred_col = find_column(preds, ["predicted_price", "predicted", "prediction", "y_pred", "pred", "forecast"])
if not pred_col:
    raise ValueError(f"Could not find Prediction column in {PRED_PATH.name}. Columns: {preds.columns.tolist()}")

preds = preds.rename(columns={pred_col: "predicted_price"})
preds["predicted_price"] = pd.to_numeric(preds["predicted_price"], errors="coerce")

# --- C. Align Predictions with Dates ---
# If the predictions file magically has a date column, merge on it.
if "date" in preds.columns or "Date" in preds.columns:
    pred_date_col = "date" if "date" in preds.columns else "Date"
    preds["date"] = pd.to_datetime(preds[pred_date_col])
    prices = pd.merge(preds[["date", "predicted_price"]], features, on="date", how="inner")
else:
    # If no date in predictions, assume it aligns with the TAIL of the features file (standard test split)
    print("[info] Predictions file has no dates. Aligning by row index (assuming test set is the last rows).")
    if len(preds) > len(features):
        raise ValueError("Predictions file has more rows than the features file!")
    
    # Take the last N rows of features to match predictions
    test_features = features.tail(len(preds)).reset_index(drop=True)
    prices = pd.concat([test_features, preds[["predicted_price"]]], axis=1)

# ============================================================
# 3. Load Events and Merge
# ============================================================
events = pd.read_csv(EVENTS_PATH)
events.columns = [str(c).strip().lstrip("\ufeff") for c in events.columns]
events["date"] = pd.to_datetime(events["date"])

events = events[(events["date"] >= prices["date"].min()) & (events["date"] <= prices["date"].max())]

events_per_day = (
    events.groupby("date")["title"]
    .apply(lambda s: "; ".join(sorted({str(x) for x in s if pd.notna(x)})))
    .reset_index()
    .rename(columns={"title": "festival_title"})
)

MAJOR = {
    "Diwali", "Holi", "Dussehra", "Vijayadashami", "Maha Shivaratri", "Ganesh Chaturthi", 
    "Navratri Begins", "Raksha Bandhan", "Rakhi", "Janmashtami *Smarta", "Krishna Janmashtami", 
    "Rama Navami", "Hanuman Jayanti", "Akshaya Tritiya", "Dhanteras", "Lakshmi Puja", "Makar Sankranti", 
    "Makara Sankranti", "Pongal", "Onam", "Chhath Puja", "Karwa Chauth", "Teej", "Hartalika Teej", 
    "Gudi Padwa", "Ugadi", "Basant Panchami", "Vasant Panchami", "Guru Purnima", "Nag Panchami", 
    "Rath Yatra", "Jagannathy Rathyatra", "Jagannath Rathyatra"
}

events_per_day["is_major"] = events_per_day["festival_title"].apply(lambda x: any(m in x for m in MAJOR))

df = prices.merge(events_per_day, on="date", how="left").sort_values("date").reset_index(drop=True)
df["is_event"] = df["festival_title"].notna()
df["is_major_event"] = df["is_major"].fillna(False).astype(bool)

# ============================================================
# 4. Spike Detection
# ============================================================
df = df.set_index("date")
df["baseline"] = df["predicted_price"].rolling(30, center=True, min_periods=5).median()
df = df.reset_index()

# ============================================================
# 5. Plotting
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
# 6. Export CSV
# ============================================================
print("\n" + "="*60)
print("GENERATING FESTIVAL PRICES CSV")
print("="*60)

# Filter only festival days
festival_days = df[df["is_event"]].copy()

# Prepare exact columns
final_csv = festival_days[["date", "festival_title", "original_price", "predicted_price"]].copy()
final_csv = final_csv.rename(columns={
    "date": "Date",
    "festival_title": "Festival_Event",
    "original_price": "Real_Price",
    "predicted_price": "Predicted_Price"
}).sort_values("Date")

# Drop any rows where we don't have a real price (just in case)
final_csv = final_csv.dropna(subset=["Real_Price"])

# Print top 20 to console so you can see it immediately
print(final_csv.head(20).to_string(index=False))
print(f"... and {len(final_csv) - 20} more rows.")

# Save to the exact requested path
final_csv.to_csv(OUT_CSV, index=False)
print(f"\n[SUCCESS] CSV saved to exactly: {OUT_CSV}")