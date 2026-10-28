
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ============================================================
# 1. INPUT AND OUTPUT PATHS
# ============================================================

input_file = (
    r"C:\Users\Ghanavi Arun\Floriculture\code"
    r"\feature_engineering\festival_auspicious_window_features.csv"
)

output_folder = (
    r"C:\Users\Ghanavi Arun\Floriculture\code"
    r"\ml_training\model5"
)

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

# Standardize column names by removing accidental spaces
df.columns = df.columns.str.strip()

required_columns = [
    "Date",
    "Price",
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
    "is_festival",
    "is_auspicious",
    "Festival_1_Day_Before",
    "Festival_1_Day_After",
    "Festival_2_Day_Before",
    "Festival_2_Day_After",
    "Festival_3_Day_Before",
    "Festival_3_Day_After",
    "Auspicious_1_Day_Before",
    "Auspicious_1_Day_After",
    "Auspicious_2_Day_Before",
    "Auspicious_2_Day_After",
    "Auspicious_3_Day_Before",
    "Auspicious_3_Day_After",
    "Festival_Within_3_Days",
    "Auspicious_Within_3_Days",
    "Any_Event_Within_3_Days"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "The input dataset is missing these required columns:\n"
        + "\n".join(missing_columns)
    )

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df = df.dropna(subset=["Date", "Price"]).copy()
df = df.sort_values("Date").reset_index(drop=True)

if df["Date"].duplicated().any():
    raise ValueError(
        "Duplicate dates found. Please resolve them before training."
    )


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

# Baseline price-history features
price_features = [
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
    "price_pct_change_1d"
]

# Exact-day event features
exact_day_features = [
    "is_festival",
    "is_auspicious"
]

# Festival window features
festival_window_features = [
    "Festival_1_Day_Before",
    "Festival_1_Day_After",
    "Festival_2_Day_Before",
    "Festival_2_Day_After",
    "Festival_3_Day_Before",
    "Festival_3_Day_After"
]

# Auspicious-day window features
auspicious_window_features = [
    "Auspicious_1_Day_Before",
    "Auspicious_1_Day_After",
    "Auspicious_2_Day_Before",
    "Auspicious_2_Day_After",
    "Auspicious_3_Day_Before",
    "Auspicious_3_Day_After"
]

# Combined event window features
combined_window_features = [
    "Festival_Within_3_Days",
    "Auspicious_Within_3_Days",
    "Any_Event_Within_3_Days"
]

features = (
    price_features
    + exact_day_features
    + festival_window_features
    + auspicious_window_features
    + combined_window_features
)

target = "Price"

print("Total features:", len(features))
print("\nFeatures used:")
for feature in features:
    print("-", feature)


# ============================================================
# 4. CLEAN FEATURE VALUES
# ============================================================

# Convert feature columns to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Event flags are treated as 0 when blank.
# Verify that blanks really mean "no event" in your dataset.
event_features = (
    exact_day_features
    + festival_window_features
    + auspicious_window_features
    + combined_window_features
)

for col in event_features:
    df[col] = df[col].fillna(0)

# Replace infinite values and fill remaining missing price-history
# features with zero to keep the same observations in each split.
# Check that this matches the preprocessing used in Models 1–4.
df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)
df[features] = df[features].fillna(0)


# ============================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df = df[df["Date"].dt.year < 2023].copy()

validation_df = df[
    df["Date"].dt.year == 2023
].copy()

test_df = df[
    df["Date"].dt.year.isin([2024, 2025])
].copy()

if train_df.empty or validation_df.empty or test_df.empty:
    raise ValueError(
        "One or more data splits are empty. "
        "Check the dates in your dataset."
    )

X_train = train_df[features]
y_train = train_df[target]

X_val = validation_df[features]
y_val = validation_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("\nDataset split:")
print("Training observations:", len(train_df))
print("Validation observations:", len(validation_df))
print("Test observations:", len(test_df))


# ============================================================
# 6. TRAIN RANDOM FOREST MODEL 5
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, dataset_name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Observations": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

results = []

results.append(
    evaluate_model(
        "Model 5 - Combined Event Windows",
        "Validation 2023",
        y_val,
        val_predictions
    )
)

results.append(
    evaluate_model(
        "Model 5 - Combined Event Windows",
        "Test 2024-2025",
        y_test,
        test_predictions
    )
)


# ============================================================
# 9. PERSISTENCE BENCHMARK
# ============================================================

# Predict today's price using yesterday's actual price.
# This is a benchmark, not another trained model.

for dataset_name, split_df in [
    ("Validation 2023", validation_df),
    ("Test 2024-2025", test_df)
]:
    actual = split_df[target].to_numpy()

    previous_price = (
        split_df["price_lag_1"].to_numpy()
    )

    results.append(
        evaluate_model(
            "Persistence Benchmark",
            dataset_name,
            actual,
            previous_price
        )
    )


# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(results)

metrics_file = os.path.join(
    output_folder,
    "model5_metrics.csv"
)

metrics_df.to_csv(metrics_file, index=False)

print("\nModel evaluation results:")
print(metrics_df.to_string(index=False))


# ============================================================
# 11. SAVE VALIDATION AND TEST PREDICTIONS
# ============================================================

val_output = pd.DataFrame({
    "Date": validation_df["Date"].to_numpy(),
    "Actual_Price": y_val.to_numpy(),
    "Predicted_Price": val_predictions,
    "Price_Lag_1": validation_df["price_lag_1"].to_numpy()
})

val_output["Prediction_Error"] = (
    val_output["Actual_Price"]
    - val_output["Predicted_Price"]
)

test_output = pd.DataFrame({
    "Date": test_df["Date"].to_numpy(),
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": test_predictions,
    "Price_Lag_1": test_df["price_lag_1"].to_numpy()
})

test_output["Prediction_Error"] = (
    test_output["Actual_Price"]
    - test_output["Predicted_Price"]
)

val_output.to_csv(
    os.path.join(output_folder, "model5_validation_predictions.csv"),
    index=False
)

test_output.to_csv(
    os.path.join(output_folder, "model5_test_predictions.csv"),
    index=False
)


# ============================================================
# 12. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(output_folder, "model5_feature_importance.csv"),
    index=False
)

print("\nTop 10 feature importances:")
print(importance_df.head(10).to_string(index=False))


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    os.path.join(output_folder, "model5_random_forest.pkl")
)


# ============================================================
# 14. COMPLETION MESSAGE
# ============================================================

print("\nModel 5 training completed successfully.")
print("Output folder:", output_folder)
print("Metrics file:", metrics_file)
print("Feature count:", len(features))

