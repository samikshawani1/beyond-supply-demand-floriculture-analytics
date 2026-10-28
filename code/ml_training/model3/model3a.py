import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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
    r"\ml_training\model3"
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
    "Any_Event_Within_3_Days",
    # Calendar source columns required for cyclical encoding
    "Month",
    "Day_Of_Week",
    "Day_Of_Year",
    "Tithi",
    "Masa",
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
# 2.5. LONG-HORIZON ROLLING MEANS
# ============================================================
# Computed here (not upstream) because the upstream file lacks
# them. Excludes today's price via .shift(1) so no leakage.
# Caveat: uses row-based windows. If the daily series has gaps,
# 90 rows ≠ 90 calendar days.

df["rolling_mean_30"] = (
    df["Price"].shift(1).rolling(30, min_periods=30).mean()
)
df["rolling_mean_90"] = (
    df["Price"].shift(1).rolling(90, min_periods=90).mean()
)

# ============================================================
# 3. DROP THE FIRST 14 ROWS (lag warm-up period)
# ============================================================
# The first 14 rows contain NaN in price_lag_14 / rolling_*_14.
# Dropping them keeps all remaining observations valid without
# relying on fillna(0), which would teach the model that prices
# were 0 a week ago.

WARMUP_ROWS = 90

if len(df) <= WARMUP_ROWS:
    raise ValueError(
        "Dataset has too few rows after warm-up removal."
    )

df = df.iloc[WARMUP_ROWS:].reset_index(drop=True)


# ============================================================
# 4. CALENDAR FEATURES WITH CYCLICAL ENCODING
# ============================================================
# Tithi is ASCII ("Krishna Dvadashi"). Masa uses Sanskrit
# transliteration with diacritics ("Bhādrapada", "Jyeṣṭha").
# Both are normalized via NFKD + combining-mark stripping
# before dictionary lookup.
#
# NOTE: For Model 3 (festival-window feature set), calendar
# features are computed here but NOT fed to the model. Kept in
# the file so reverting to Model 5 is a one-line change in §5.

import unicodedata


def strip_diacritics(s):
    """Remove combining diacritics: 'Jyeṣṭha' -> 'jyestha'."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(
        c for c in nfkd if not unicodedata.combining(c)
    )


# --- Tithi name -> 1-14 ordinal (Shukla paksha) ------------
_TITHI_BASE = {
    "pratipada": 1, "prathama": 1, "pratipad": 1, "padyami": 1,
    "dwitiya": 2, "dvitiya": 2, "vidiya": 2,
    "tritiya": 3, "thritiya": 3, "thadiya": 3,
    "chaturthi": 4, "chavithi": 4,
    "panchami": 5,
    "shashthi": 6, "shashti": 6, "sashti": 6,
    "saptami": 7,
    "ashtami": 8,
    "navami": 9,
    "dashami": 10, "dasami": 10,
    "ekadashi": 11, "ekadasi": 11,
    "dwadashi": 12, "dvadasi": 12, "dvadashi": 12,
    "trayodashi": 13, "trayodasi": 13,
    "chaturdashi": 14, "chaturdasi": 14,
}

# --- Masa name -> 1-12 ordinal -----------------------------
# Keys are in diacritic-stripped lowercase form. Covers the
# exact spellings present in this dataset plus common variants.
_MASA_ORDINAL = {
    "chaitra": 1, "caitra": 1, "chaitram": 1,
    "vaishakha": 2, "vaisakha": 2, "vaishakham": 2,
    "jyeshtha": 3, "jyeshta": 3, "jyestha": 3,
    "jyesta": 3, "jyeshtham": 3,
    "ashadha": 4, "asadha": 4, "ashadam": 4,
    "shravana": 5, "sravana": 5,
    "shravanam": 5, "sravanam": 5,
    "bhadrapada": 6, "bhadrapad": 6, "bhadra": 6,
    "ashwin": 7, "ashwina": 7, "aswin": 7,
    "asvina": 7, "aswayuja": 7,
    "kartika": 8, "karthika": 8, "karttika": 8,
    "karthik": 8, "kartik": 8,
    "margashirsha": 9, "margashirsh": 9, "margashira": 9,
    "margasira": 9, "margasirsa": 9,
    "agrahayana": 9, "agrahayan": 9,
    "pausha": 10, "pushya": 10, "paush": 10,
    "paushama": 10, "pausa": 10,
    "magha": 11, "magh": 11, "magham": 11,
    "phalguna": 12, "phalgun": 12, "palguna": 12,
}


def _try_numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _detect_paksha(s):
    if "krishna" in s or "waning" in s:
        return "krishna"
    if "shukla" in s or "waxing" in s:
        return "shukla"
    stripped = s.strip()
    if stripped.startswith(("k.", "k ", "k-", "k:")):
        return "krishna"
    if stripped.startswith(("s.", "s ", "s-", "s:")):
        return "shukla"
    return None


def parse_tithi(value):
    if pd.isna(value):
        return np.nan
    numeric = _try_numeric(value)
    if not np.isnan(numeric):
        return numeric
    s = strip_diacritics(str(value)).strip().lower()
    if "purnima" in s or "paurnima" in s or "poornima" in s:
        return 15.0
    if "amavasya" in s or "amavasi" in s or "amavas" in s:
        return 30.0
    paksha = _detect_paksha(s)
    for name, num in _TITHI_BASE.items():
        if name in s:
            return float(15 + num) if paksha == "krishna" else float(num)
    return np.nan


def parse_masa(value):
    if pd.isna(value):
        return np.nan
    numeric = _try_numeric(value)
    if not np.isnan(numeric):
        return numeric
    s = strip_diacritics(str(value)).strip().lower()
    for name, num in _MASA_ORDINAL.items():
        if name in s:
            return float(num)
    return np.nan


def _encode_cyclical(values, period):
    arr = values.astype(float)
    unknown = arr.isna()
    angle = 2.0 * np.pi * arr / period
    sin_vals = np.sin(angle).fillna(0.0)
    cos_vals = np.cos(angle).fillna(0.0)
    return sin_vals, cos_vals, unknown.astype(int)


# Parse to ordinals
df["Tithi_num"] = df["Tithi"].apply(parse_tithi)
df["Masa_num"] = df["Masa"].apply(parse_masa)

tithi_unknown = int(df["Tithi_num"].isna().sum())
masa_unknown = int(df["Masa_num"].isna().sum())
print(
    f"Tithi unparseable: {tithi_unknown} of {len(df)} "
    f"({100 * tithi_unknown / len(df):.1f}%)"
)
print(
    f"Masa unparseable:  {masa_unknown} of {len(df)} "
    f"({100 * masa_unknown / len(df):.1f}%)"
)

if tithi_unknown > 0:
    print("\nSample unparseable Tithi values:")
    print(df.loc[df["Tithi_num"].isna(), "Tithi"]
            .value_counts(dropna=False).head(10))
if masa_unknown > 0:
    print("\nSample unparseable Masa values:")
    print(df.loc[df["Masa_num"].isna(), "Masa"]
            .value_counts(dropna=False).head(10))

# Cyclical encoding
df["Month_sin"], df["Month_cos"], _ = _encode_cyclical(df["Month"], 12)
df["Dow_sin"], df["Dow_cos"], _ = _encode_cyclical(df["Day_Of_Week"], 7)
df["Doy_sin"], df["Doy_cos"], _ = _encode_cyclical(df["Day_Of_Year"], 365.25)
(
    df["Tithi_sin"], df["Tithi_cos"], df["Tithi_unknown"]
) = _encode_cyclical(df["Tithi_num"], 30)
(
    df["Masa_sin"], df["Masa_cos"], df["Masa_unknown"]
) = _encode_cyclical(df["Masa_num"], 12)

# Additional Fourier harmonics on day-of-year
for k in (2, 3):
    df[f"Doy_{k}x_sin"] = np.sin(
        2.0 * np.pi * k * df["Day_Of_Year"].astype(float) / 365.25
    )
    df[f"Doy_{k}x_cos"] = np.cos(
        2.0 * np.pi * k * df["Day_Of_Year"].astype(float) / 365.25
    )


# ============================================================
# 5. DEFINE FEATURES
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
    "price_pct_change_1d",
    "rolling_mean_30",
    "rolling_mean_90",
]

# Exact-day event features
exact_day_features = [
    "is_festival",
    "is_auspicious",
]

# Festival window features
festival_window_features = [
    "Festival_1_Day_Before",
    "Festival_1_Day_After",
    "Festival_2_Day_Before",
    "Festival_2_Day_After",
    "Festival_3_Day_Before",
    "Festival_3_Day_After",
]

# Auspicious-day window features (NOT used in Model 3)
auspicious_window_features = [
    "Auspicious_1_Day_Before",
    "Auspicious_1_Day_After",
    "Auspicious_2_Day_Before",
    "Auspicious_2_Day_After",
    "Auspicious_3_Day_Before",
    "Auspicious_3_Day_After",
]

# Combined event window features (NOT used in Model 3)
combined_window_features = [
    "Festival_Within_3_Days",
    "Auspicious_Within_3_Days",
    "Any_Event_Within_3_Days",
]

# Calendar features (cyclical + Fourier) — NOT used in Model 3
calendar_features = [
    "Month_sin", "Month_cos",
    "Dow_sin", "Dow_cos",
    "Doy_sin", "Doy_cos",
    "Doy_2x_sin", "Doy_2x_cos",
    "Doy_3x_sin", "Doy_3x_cos",
    "Tithi_sin", "Tithi_cos",
    "Masa_sin", "Masa_cos",
]

# ============================================================
# MODEL 3 FEATURE SET
# ============================================================
# Price features + exact-day flags + 6 festival-window columns.
# Auspicious-window, combined-window, and calendar features are
# deliberately excluded. To revert to Model 5, add the other
# three groups back to the tuple below.

features = (
    price_features
    + exact_day_features
    + festival_window_features
)

target = "Price"

print("Total features:", len(features))
print("\nFeatures used:")
for feature in features:
    print("-", feature)


# ============================================================
# 6. CLEAN FEATURE VALUES
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

# Replace infinite values from price_pct_change_1d
df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)

# After dropping the first 14 rows, price-history features
# should contain no NaNs. Drop any stragglers (e.g. internal
# data gaps) instead of fillna(0).
price_nan_rows = df[price_features].isna().any(axis=1).sum()
if price_nan_rows > 0:
    print(
        f"Warning: dropping {price_nan_rows} rows with remaining "
        f"NaN in price-history features."
    )
    df = df.dropna(subset=price_features).reset_index(drop=True)

# Calendar features should never be NaN, but guard anyway.
# (Model 3 doesn't use them; this is a no-op safety net.)
calendar_nan_rows = df[calendar_features].isna().any(axis=1).sum()
if calendar_nan_rows > 0:
    print(
        f"Warning: dropping {calendar_nan_rows} rows with NaN "
        f"in calendar features."
    )
    df = df.dropna(subset=calendar_features).reset_index(drop=True)


# ============================================================
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================
# TimeSeriesSplit cross-validation is applied to all data before
# 2024. The 2024-2025 period is kept as a final untouched test.

cv_df = df[df["Date"].dt.year < 2024].copy()
test_df = df[df["Date"].dt.year.isin([2024, 2025])].copy()

if cv_df.empty or test_df.empty:
    raise ValueError(
        "One or more data splits is empty. "
        "Check the dates in your dataset."
    )

print("\nDataset split:")
print("CV (pre-2024) observations:", len(cv_df))
print("Test 2024-2025 observations:", len(test_df))


# ============================================================
# 8. TRAIN RANDOM FOREST WITH TimeSeriesSplit CV
# ============================================================
# Target is modelled as log return from yesterday so that
# persistence becomes the zero point of the target space and
# no extrapolation is required. Predictions are reconstructed
# as price_lag_1 * exp(predicted_return).

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1,
)

N_SPLITS = 5

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

X_cv_all = cv_df[features]
# Target = log return from yesterday. Predicting 0 = persistence.
y_cv_all_log_return = (
    np.log(cv_df[target]) - np.log(cv_df["price_lag_1"])
)
y_cv_all_log_return = (
    y_cv_all_log_return.replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
y_cv_all_actual = cv_df[target].to_numpy()

cv_predictions_records = []
cv_metrics = []

for fold_idx, (tr_idx, vl_idx) in enumerate(
    tscv.split(X_cv_all), start=1
):
    X_tr = X_cv_all.iloc[tr_idx]
    y_tr = y_cv_all_log_return.iloc[tr_idx]

    X_vl = X_cv_all.iloc[vl_idx]
    y_vl_actual = y_cv_all_actual[vl_idx]

    fold_model = RandomForestRegressor(**RF_PARAMS)
    fold_model.fit(X_tr, y_tr)

    y_vl_pred_log_return = fold_model.predict(X_vl)
    y_vl_pred = (
        cv_df["price_lag_1"].iloc[vl_idx].to_numpy()
        * np.exp(y_vl_pred_log_return)
    )

    persistence_pred = (
        cv_df["price_lag_1"].iloc[vl_idx].to_numpy()
    )

    cv_metrics.append({
        "Model": "Model 3 - Festival Windows",
        "Dataset": f"CV Fold {fold_idx}",
        "Observations": len(vl_idx),
        "MAE": mean_absolute_error(y_vl_actual, y_vl_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_vl_actual, y_vl_pred)
        ),
        "R2": r2_score(y_vl_actual, y_vl_pred),
        "Persistence_MAE": mean_absolute_error(
            y_vl_actual, persistence_pred
        ),
        "Persistence_RMSE": np.sqrt(
            mean_squared_error(y_vl_actual, persistence_pred)
        ),
    })

    cv_predictions_records.append(pd.DataFrame({
        "Date": cv_df["Date"].iloc[vl_idx].to_numpy(),
        "Fold": fold_idx,
        "Actual_Price": y_vl_actual,
        "Predicted_Price": y_vl_pred,
        "Price_Lag_1": cv_df["price_lag_1"].iloc[vl_idx].to_numpy(),
    }))

cv_predictions_df = pd.concat(
    cv_predictions_records, ignore_index=True
)

cv_predictions_df["Prediction_Error"] = (
    cv_predictions_df["Actual_Price"]
    - cv_predictions_df["Predicted_Price"]
)

cv_metrics_df = pd.DataFrame(cv_metrics)

cv_mean_row = {
    "Model": "Model 3 - Festival Windows",
    "Dataset": f"CV Mean ({N_SPLITS} folds)",
    "Observations": int(cv_metrics_df["Observations"].sum()),
    "MAE": cv_metrics_df["MAE"].mean(),
    "RMSE": cv_metrics_df["RMSE"].mean(),
    "R2": cv_metrics_df["R2"].mean(),
    "Persistence_MAE": cv_metrics_df["Persistence_MAE"].mean(),
    "Persistence_RMSE": cv_metrics_df["Persistence_RMSE"].mean(),
}


# ============================================================
# 9. TRAIN FINAL MODEL ON ALL PRE-2024 DATA
# ============================================================

X_train_full = cv_df[features]
y_train_full_log_return = (
    np.log(cv_df[target]) - np.log(cv_df["price_lag_1"])
).replace([np.inf, -np.inf], np.nan).fillna(0.0)

model = RandomForestRegressor(**RF_PARAMS)
model.fit(X_train_full, y_train_full_log_return)


# ============================================================
# 10. EVALUATION FUNCTION
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
# 11. GENERATE TEST PREDICTIONS
# ============================================================

X_test = test_df[features]
y_test = test_df[target].to_numpy()

test_log_return_pred = model.predict(X_test)
test_predictions = (
    test_df["price_lag_1"].to_numpy() * np.exp(test_log_return_pred)
)

results = [cv_mean_row] + cv_metrics

results.append(
    evaluate_model(
        "Model 3 - Festival Windows",
        "Test 2024-2025",
        y_test,
        test_predictions
    )
)

results.append({
    "Model": "Persistence Benchmark",
    "Dataset": "Test 2024-2025",
    "Observations": len(y_test),
    "MAE": mean_absolute_error(
        y_test, test_df["price_lag_1"].to_numpy()
    ),
    "RMSE": np.sqrt(
        mean_squared_error(
            y_test, test_df["price_lag_1"].to_numpy()
        )
    ),
    "R2": r2_score(
        y_test, test_df["price_lag_1"].to_numpy()
    ),
})


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(results)

metrics_file = os.path.join(
    output_folder,
    "model3_metrics.csv"
)

metrics_df.to_csv(metrics_file, index=False)

print("\nModel evaluation results:")
print(metrics_df.to_string(index=False))


# ============================================================
# 14. SAVE CV AND TEST PREDICTIONS
# ============================================================

cv_predictions_df.to_csv(
    os.path.join(output_folder, "model3_cv_predictions.csv"),
    index=False
)

test_output = pd.DataFrame({
    "Date": test_df["Date"].to_numpy(),
    "Actual_Price": y_test,
    "Predicted_Price": test_predictions,
    "Price_Lag_1": test_df["price_lag_1"].to_numpy()
})

test_output["Prediction_Error"] = (
    test_output["Actual_Price"]
    - test_output["Predicted_Price"]
)

bias = test_output["Prediction_Error"].mean()
print(
    f"\nTest mean signed error (actual - predicted): {bias:+.1f}"
)
print(f"Test mean actual:    {test_output['Actual_Price'].mean():.1f}")
print(f"Test mean predicted: {test_output['Predicted_Price'].mean():.1f}")

test_output.to_csv(
    os.path.join(output_folder, "model3_test_predictions.csv"),
    index=False
)

# ============================================================
# 14.5. FESTIVAL / AUSPICIOUS DAY DIAGNOSTIC
# ============================================================
# Does the model beat persistence specifically on the days
# the event features are supposed to help?

merged = test_output.merge(
    test_df[["Date", "is_festival", "is_auspicious"]],
    on="Date", how="left"
)
merged["Model_Abs_Error"] = merged["Prediction_Error"].abs()
merged["Persistence_Abs_Error"] = (
    merged["Actual_Price"] - merged["Price_Lag_1"]
).abs()

print("\nTest MAE by is_festival:")
print(
    merged.groupby("is_festival")[
        ["Model_Abs_Error", "Persistence_Abs_Error"]
    ].agg(["mean", "count"]).round(2)
)
print("\nTest MAE by is_auspicious:")
print(
    merged.groupby("is_auspicious")[
        ["Model_Abs_Error", "Persistence_Abs_Error"]
    ].agg(["mean", "count"]).round(2)
)


# ============================================================
# 15. SAVE FEATURE IMPORTANCE
# ============================================================
# Importances reflect contribution to explaining variance in
# log return, not raw Price. Ranking is still meaningful.

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(output_folder, "model3_feature_importance.csv"),
    index=False
)

print("\nTop 10 feature importances:")
print(importance_df.head(10).to_string(index=False))

imp = importance_df.set_index("Feature")["Importance"]

price_imp = imp.reindex(price_features).sum()
event_imp = imp.reindex(
    exact_day_features
    + festival_window_features
).sum()
# Calendar and auspicious groups are absent in Model 3;
# their importance will print as 0.000, which is correct.
calendar_imp = 0.0

print("\nImportance by group:")
print(f"  Price history:  {price_imp:.3f}")
print(f"  Event flags:    {event_imp:.3f}")
print(f"  Calendar:       {calendar_imp:.3f} (not used)")
print(f"  Top single:     {imp.idxmax()} = {imp.max():.3f}")

# ============================================================
# 16. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    os.path.join(output_folder, "model3_random_forest.pkl")
)


# ============================================================
# 17. COMPLETION MESSAGE
# ============================================================

print("\nModel 3 training completed successfully.")
print("Output folder:", output_folder)
print("Metrics file:", metrics_file)
print("Feature count:", len(features))