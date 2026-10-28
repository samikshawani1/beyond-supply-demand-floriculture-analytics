
import os
import numpy as np
import pandas as pd
import joblib

# ---------------- PATHS ----------------
input_file = r"C:\Users\Ghanavi Arun\Floriculture\code\feature_engineering\festival_auspicious_window_features.csv"

model_file = r"C:\Users\Ghanavi Arun\Floriculture\code\ml_training\model5\model5_random_forest.pkl"

output_folder = r"C:\Users\Ghanavi Arun\Floriculture\code\ml_training\model5\model5prediction"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder, "chrysanthemum_prices_2026.csv"
)

# ---------------- LOAD HISTORICAL DATA ----------------
df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df = df.dropna(subset=["Date", "Price"])
df["Date"] = df["Date"].dt.normalize()
df = df.sort_values("Date").drop_duplicates("Date", keep="last")

if df.empty:
    raise ValueError("No valid historical Date and Price data found.")

# ---------------- LOAD TRAINED MODEL ----------------
model = joblib.load(model_file)

features = [
    "price_lag_1", "price_lag_2", "price_lag_3",
    "price_lag_7", "price_lag_14",
    "rolling_mean_3", "rolling_mean_7", "rolling_mean_14",
    "rolling_std_7", "price_change_1d", "price_pct_change_1d",
    "is_festival", "is_auspicious",
    "Festival_1_Day_Before", "Festival_1_Day_After",
    "Festival_2_Day_Before", "Festival_2_Day_After",
    "Festival_3_Day_Before", "Festival_3_Day_After",
    "Auspicious_1_Day_Before", "Auspicious_1_Day_After",
    "Auspicious_2_Day_Before", "Auspicious_2_Day_After",
    "Auspicious_3_Day_Before", "Auspicious_3_Day_After",
    "Festival_Within_3_Days", "Auspicious_Within_3_Days",
    "Any_Event_Within_3_Days"
]

# Confirm that the loaded model expects the same features
if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != features:
        raise ValueError(
            "The feature list does not match the trained model. "
            "Check the feature order used during training."
        )

# Historical prices are used to initialize recursive forecasting
prices = df["Price"].astype(float).tolist()

# ---------------- FORECAST ALL OF 2026 ----------------
dates_2026 = pd.date_range(
    start="2026-01-01",
    end="2026-12-31",
    freq="D"
)

results = []

for current_date in dates_2026:

    # All these values use only previous actual or predicted prices
    s = pd.Series(prices, dtype=float)

    if len(s) < 14:
        raise ValueError("At least 14 historical prices are required.")

    previous_price = s.iloc[-1]
    price_two_days_ago = s.iloc[-2]

    row = {
        "price_lag_1": s.iloc[-1],
        "price_lag_2": s.iloc[-2],
        "price_lag_3": s.iloc[-3],
        "price_lag_7": s.iloc[-7],
        "price_lag_14": s.iloc[-14],

        "rolling_mean_3": s.iloc[-3:].mean(),
        "rolling_mean_7": s.iloc[-7:].mean(),
        "rolling_mean_14": s.iloc[-14:].mean(),
        "rolling_std_7": s.iloc[-7:].std(),

        "price_change_1d": previous_price - price_two_days_ago,

        "price_pct_change_1d": (
            (previous_price - price_two_days_ago)
            / price_two_days_ago * 100
            if price_two_days_ago != 0 else 0
        ),

        # No 2026 festival calendar is supplied.
        # These event features are assumed to be zero.
        "is_festival": 0,
        "is_auspicious": 0,

        "Festival_1_Day_Before": 0,
        "Festival_1_Day_After": 0,
        "Festival_2_Day_Before": 0,
        "Festival_2_Day_After": 0,
        "Festival_3_Day_Before": 0,
        "Festival_3_Day_After": 0,

        "Auspicious_1_Day_Before": 0,
        "Auspicious_1_Day_After": 0,
        "Auspicious_2_Day_Before": 0,
        "Auspicious_2_Day_After": 0,
        "Auspicious_3_Day_Before": 0,
        "Auspicious_3_Day_After": 0,

        "Festival_Within_3_Days": 0,
        "Auspicious_Within_3_Days": 0,
        "Any_Event_Within_3_Days": 0
    }

    X = pd.DataFrame([row], columns=features)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    predicted_price = float(model.predict(X)[0])

    # Use this prediction to calculate the following day's features
    prices.append(predicted_price)

    results.append({
        "Date": current_date.strftime("%Y-%m-%d"),
        "Predicted_Price": round(predicted_price, 2)
    })

# ---------------- SAVE CSV ----------------
forecast_df = pd.DataFrame(results)
forecast_df.to_csv(output_file, index=False)

print("2026 forecasting completed.")
print("Total dates predicted:", len(forecast_df))
print("Output file:", output_file)
print("\nFirst five predictions:")
print(forecast_df.head())
print("\nLast five predictions:")
print(forecast_df.tail())
