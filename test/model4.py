


import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. INPUT AND OUTPUT PATHS
# ============================================================

input_file = (
    r"C:\Users\Ghanavi Arun\Floriculture\code"
    r"\feature_engineering\festival_auspicious_window_features.csv"
)

output_folder = (
    r"C:\Users\Ghanavi Arun\Floriculture\code"
    r"\ml_training\model4"
)

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Input CSV not found:\n{input_file}"
    )

df = pd.read_csv(input_file)

required_columns = [
    "Date",
    "Price",
    "is_festival",
    "is_auspicious",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")


# ============================================================
# 3. PREPARE EXACT-DAY EVENT FLAGS
# ============================================================

def convert_event_flag(series, column_name):
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
    }

    values = series.astype("string").str.strip().str.lower()
    converted = values.map(mapping)

    # Assumes missing flags mean no event.
    # Confirm this assumption is valid for your dataset.
    converted = converted.fillna(0).astype("int8")

    return converted


df["is_festival"] = convert_event_flag(
    df["is_festival"], "is_festival"
)

df["is_auspicious"] = convert_event_flag(
    df["is_auspicious"], "is_auspicious"
)


# ============================================================
# 4. SORT AND CHECK DATES
# ============================================================

df = df.sort_values("Date").reset_index(drop=True)

if df["Date"].isna().any():
    raise ValueError("Some rows contain invalid dates.")

if df["Date"].duplicated().any():
    raise ValueError(
        "Duplicate dates found. Resolve them before training."
    )


# ============================================================
# 5. SELECT FEATURES FOR MODEL 4
# ============================================================

# Price-history features
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
    "price_pct_change_1d",
]

# Exact-day event flags
exact_day_features = [
    "is_festival",
    "is_auspicious",
]

# Auspicious-day windows only
auspicious_window_features = [
    "Auspicious_1_Day_Before",
    "Auspicious_1_Day_After",
    "Auspicious_2_Day_Before",
    "Auspicious_2_Day_After",
    "Auspicious_3_Day_Before",
    "Auspicious_3_Day_After",
]

feature_columns = (
    price_features
    + exact_day_features
    + auspicious_window_features
)

missing_features = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing Model 4 features: {missing_features}"
    )

target_column = "Price"


# ============================================================
# 6. PREPARE MODELING DATA
# ============================================================

# The original CSV is not modified.
# Rows with missing target/features are excluded from training.

model_df = df.dropna(
    subset=["Date", target_column] + feature_columns
).copy()

X = model_df[feature_columns]
y = model_df[target_column]
dates = model_df["Date"]


# ============================================================
# 7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLITS
# ============================================================

train_mask = dates < "2023-01-01"

validation_mask = (
    (dates >= "2023-01-01")
    & (dates < "2024-01-01")
)

test_mask = (
    (dates >= "2024-01-01")
    & (dates < "2026-01-01")
)

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[validation_mask]
y_val = y.loc[validation_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError(
        "A train, validation, or test split is empty. "
        "Check the dates in the input CSV."
    )

print("\nDATASET SPLIT")
print(f"Training rows (before 2023): {len(X_train)}")
print(f"Validation rows (2023):      {len(X_val)}")
print(f"Test rows (2024-2025):        {len(X_test)}")


# ============================================================
# 8. TRAIN RANDOM FOREST MODEL
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

print("\nModel 4 training completed.")


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

def calculate_metrics(actual, predicted, dataset_name):
    return {
        "Model": "Model 4 - Auspicious-Day Windows",
        "Dataset": dataset_name,
        "Observations": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
    }


val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

metrics = pd.DataFrame([
    calculate_metrics(
        y_val, val_predictions, "Validation 2023"
    ),
    calculate_metrics(
        y_test, test_predictions, "Test 2024-2025"
    ),
])

metrics_file = os.path.join(
    output_folder,
    "model4_metrics.csv"
)

metrics.to_csv(metrics_file, index=False)


# ============================================================
# 10. SAVE TEST PREDICTIONS
# ============================================================

test_results = pd.DataFrame({
    "Date": dates.loc[X_test.index].values,
    "Actual_Price": y_test.values,
    "Predicted_Price": test_predictions,
    "Prediction_Error": y_test.values - test_predictions,
    "Absolute_Error": np.abs(
        y_test.values - test_predictions
    ),
})

# Include auspicious-day flags for interpretation.
for col in ["is_auspicious"] + auspicious_window_features:
    test_results[col] = model_df.loc[X_test.index, col].values

predictions_file = os.path.join(
    output_folder,
    "model4_test_predictions.csv"
)

test_results.to_csv(predictions_file, index=False)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_,
})

importance_df = importance_df.sort_values(
    "Importance", ascending=False
)

importance_file = os.path.join(
    output_folder,
    "model4_feature_importance.csv"
)

importance_df.to_csv(importance_file, index=False)


# ============================================================
# 12. SAVE TRAINED MODEL AND FEATURE LIST
# ============================================================

model_file = os.path.join(
    output_folder,
    "model4_auspicious_windows.joblib"
)

joblib.dump(model, model_file)

feature_list_file = os.path.join(
    output_folder,
    "model4_feature_list.txt"
)

with open(feature_list_file, "w", encoding="utf-8") as file:
    file.write("\n".join(feature_columns))


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\nMODEL 4 EVALUATION RESULTS")
print(metrics.to_string(index=False))

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

print("\nAll files saved in:")
print(output_folder)

print("\nGenerated files:")
print("- model4_metrics.csv")
print("- model4_test_predictions.csv")
print("- model4_feature_importance.csv")
print("- model4_auspicious_windows.joblib")
print("- model4_feature_list.txt")

