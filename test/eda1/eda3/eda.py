"""
Chrysanthemum Price EDA (2010–2025)
----------------------------------
Run this script from your EDA project folder:
    python chrysanthemum_eda.py

The source CSV is read only. No rows are removed, no values are replaced,
and no cleaned dataset is written. All outputs are saved to ./eda_outputs/.

Requirements:
    pip install pandas numpy matplotlib
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------- CONFIGURATION ----------------------------
# User-provided source path. Change this if your CSV is in another location.
DATA_PATH = Path(r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\all_data_classified.csv")
OUTPUT_DIR = Path.cwd() / "eda_outputs"

DATE_COL = "Date"
PRICE_COL = "Price"
TITHI_COL = "Tithi"
MASA_COL = "Masa"
EVENT_COL = "Events"

# IMPORTANT:
# Your file has Tithi and Events columns, but no explicit "Auspicious" flag.
# This list is therefore a configurable analysis definition, not a claim
# that these are the only auspicious tithis. Edit it to match your project/domain.
AUSPICIOUS_TITHIS = {
    "Shukla Ekadashi",
    "Shukla Dvadashi",
    "Shukla Trayodashi",
    "Shukla Purnima",
}

START_YEAR = 2010
END_YEAR = 2025
DPI = 160
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_plot(fig, filename):
    """Save a figure and close it to keep memory use low."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def money_axis(ax, axis="y"):
    """Use readable price labels without assuming a currency symbol."""
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{DATA_PATH}\n\n"
            "Edit DATA_PATH near the top of chrysanthemum_eda.py to point "
            "to your all_data.csv file."
        )

    # Read source file. The source file is never written to.
    raw = pd.read_csv(DATA_PATH)
    required = {DATE_COL, PRICE_COL, TITHI_COL, MASA_COL, EVENT_COL}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Create a separate analysis frame; do not alter or overwrite raw.
    data = raw.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data[PRICE_COL] = pd.to_numeric(data[PRICE_COL], errors="coerce")

    # ---------------------------- DATA DESCRIPTION ----------------------------
    description = pd.DataFrame({
        "column": raw.columns,
        "dtype_in_source": [str(raw[c].dtype) for c in raw.columns],
        "non_null_count": [raw[c].notna().sum() for c in raw.columns],
        "missing_count": [raw[c].isna().sum() for c in raw.columns],
        "missing_percent": [(raw[c].isna().mean() * 100).round(2) for c in raw.columns],
        "unique_values": [raw[c].nunique(dropna=True) for c in raw.columns],
    })
    description.to_csv(OUTPUT_DIR / "01_column_description.csv", index=False)

    numeric_summary = data[PRICE_COL].describe().to_frame(name="Price").round(3)
    numeric_summary.to_csv(OUTPUT_DIR / "02_price_descriptive_statistics.csv")

    overview_lines = [
        "CHRYSANTHEMUM PRICE DATA — EDA OVERVIEW",
        "=" * 45,
        f"Rows: {len(raw):,}",
        f"Columns: {len(raw.columns)}",
        f"Date range: {data[DATE_COL].min()} to {data[DATE_COL].max()}",
        f"Duplicate full rows: {raw.duplicated().sum():,}",
        f"Duplicate dates: {data[DATE_COL].duplicated().sum():,}",
        f"Invalid/missing parsed dates: {data[DATE_COL].isna().sum():,}",
        f"Missing/non-numeric prices in analysis copy: {data[PRICE_COL].isna().sum():,}",
        "",
        "PRICE SUMMARY",
        f"Minimum: {data[PRICE_COL].min():,.2f}",
        f"Maximum: {data[PRICE_COL].max():,.2f}",
        f"Mean: {data[PRICE_COL].mean():,.2f}",
        f"Median: {data[PRICE_COL].median():,.2f}",
        f"Standard deviation: {data[PRICE_COL].std():,.2f}",
        f"Q1 (25%): {data[PRICE_COL].quantile(.25):,.2f}",
        f"Q3 (75%): {data[PRICE_COL].quantile(.75):,.2f}",
        "",
        "MISSING VALUES BY COLUMN",
        raw.isna().sum().to_string(),
        "",
        "Note: Event blanks are retained as blanks in the source. For event-vs-non-event",
        "plots only, a non-empty Events cell is treated as an event day.",
        "No cleaning or source-file modification is performed.",
    ]
    (OUTPUT_DIR / "00_eda_overview.txt").write_text("\n".join(overview_lines), encoding="utf-8")

    # Basic price histogram and boxplot
    prices = data[PRICE_COL].dropna()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(prices, bins=40, edgecolor="black", alpha=0.8)
    ax.set(title="Chrysanthemum Price Distribution", xlabel="Price", ylabel="Number of days")
    money_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "03_price_distribution_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(prices, vert=False, patch_artist=True, showmeans=True)
    ax.set(title="Chrysanthemum Prices — Overall Boxplot", xlabel="Price")
    money_axis(ax)
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, "04_overall_price_boxplot.png")

    # ---------------------------- IQR OUTLIERS ----------------------------
    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Outliers are flagged for analysis only; never removed from data.
    outlier_mask = data[PRICE_COL].notna() & (
        (data[PRICE_COL] < lower_bound) | (data[PRICE_COL] > upper_bound)
    )
    outliers = data.loc[outlier_mask].copy()
    outliers.to_csv(OUTPUT_DIR / "05_overall_iqr_outliers.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    non_outliers = data.loc[~outlier_mask & data[DATE_COL].notna()]
    ax.scatter(non_outliers[DATE_COL], non_outliers[PRICE_COL],
               s=12, alpha=0.55, label="Within IQR bounds")
    if not outliers.empty:
        ax.scatter(outliers[DATE_COL], outliers[PRICE_COL],
                   s=28, marker="x", label="IQR outlier", zorder=3)
    ax.axhline(upper_bound, linestyle="--", linewidth=1.4,
               label=f"Upper bound: {upper_bound:,.1f}")
    ax.axhline(lower_bound, linestyle=":", linewidth=1.4,
               label=f"Lower bound: {lower_bound:,.1f}")
    ax.set(title="Chrysanthemum Prices Over Time — Overall Outliers Highlighted",
           xlabel="Date", ylabel="Price")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    money_axis(ax)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    save_plot(fig, "06_overall_dataset_outliers_over_time.png")

    # Yearly outlier summary (same global IQR bounds for consistent comparison)
    yearly_outliers = []
    for year in range(START_YEAR, END_YEAR + 1):
        part = data[data[DATE_COL].dt.year == year]
        yearly_outliers.append({
            "Year": year,
            "Rows": len(part),
            "Min_price": part[PRICE_COL].min(),
            "Max_price": part[PRICE_COL].max(),
            "Mean_price": part[PRICE_COL].mean(),
            "Median_price": part[PRICE_COL].median(),
            "IQR_outlier_count_using_global_bounds": int(outlier_mask.loc[part.index].sum()),
        })
    pd.DataFrame(yearly_outliers).to_csv(OUTPUT_DIR / "07_yearly_price_summary.csv", index=False)

    # ---------------------------- MONTHLY AVERAGES ----------------------------
    # One combined chart for all years, plus one individual chart for every year.
    valid_dates = data.dropna(subset=[DATE_COL, PRICE_COL]).copy()
    valid_dates["Year"] = valid_dates[DATE_COL].dt.year
    valid_dates["Month"] = valid_dates[DATE_COL].dt.month
    monthly = (valid_dates.groupby(["Year", "Month"], as_index=False)[PRICE_COL]
               .mean().rename(columns={PRICE_COL: "Average_Price"}))
    monthly.to_csv(OUTPUT_DIR / "08_monthly_average_prices_2010_2025.csv", index=False)

    fig, ax = plt.subplots(figsize=(15, 7))
    for year, group in monthly.groupby("Year"):
        ax.plot(group["Month"], group["Average_Price"], marker="o",
                linewidth=1.5, markersize=3, label=str(year))
    ax.set(title="Monthly Average Chrysanthemum Prices by Year (2010–2025)",
           xlabel="Month", ylabel="Average price")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    money_axis(ax)
    ax.grid(alpha=0.25)
    ax.legend(title="Year", bbox_to_anchor=(1.02, 1), loc="upper left",
              ncol=1, frameon=False)
    save_plot(fig, "09_monthly_average_all_years_comparison.png")

    for year in range(START_YEAR, END_YEAR + 1):
        group = monthly[monthly["Year"] == year].sort_values("Month")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(group["Month"], group["Average_Price"], marker="o", linewidth=2)
        ax.set(title=f"Monthly Average Chrysanthemum Price — {year}",
               xlabel="Month", ylabel="Average price")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        money_axis(ax)
        ax.grid(alpha=0.25)
        save_plot(fig, f"monthly_average_{year}.png")

    # ---------------------------- EVENT / AUSPICIOUS DAY ANALYSIS ----------------------------
    # Event day = Events cell contains at least one non-whitespace value.
    # Auspicious day = Tithi exactly matches AUSPICIOUS_TITHIS configured above.
    data["Is_Event_Day"] = data[EVENT_COL].fillna("").astype(str).str.strip().ne("")
    data["Is_Auspicious_Tithi"] = data[TITHI_COL].fillna("").astype(str).str.strip().isin(
        AUSPICIOUS_TITHIS
    )

    # Save price comparisons without altering the original data.
    comparison = data.groupby(["Is_Event_Day", "Is_Auspicious_Tithi"])[PRICE_COL].agg(
        ["count", "mean", "median", "min", "max", "std"]
    ).reset_index()
    comparison.to_csv(OUTPUT_DIR / "10_event_auspicious_price_comparison.csv", index=False)

    # Year-by-year graph: daily prices with event days and auspicious-tithi days highlighted.
    for year in range(START_YEAR, END_YEAR + 1):
        part = data[data[DATE_COL].dt.year == year].dropna(subset=[DATE_COL, PRICE_COL]).copy()
        if part.empty:
            continue
        event_part = part[part["Is_Event_Day"]]
        ausp_part = part[part["Is_Auspicious_Tithi"]]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(part[DATE_COL], part[PRICE_COL], linewidth=1, alpha=0.65,
                label="Daily chrysanthemum price")
        if not event_part.empty:
            ax.scatter(event_part[DATE_COL], event_part[PRICE_COL], s=35,
                       marker="s", label="Festival / event day", zorder=3)
        if not ausp_part.empty:
            ax.scatter(ausp_part[DATE_COL], ausp_part[PRICE_COL], s=42,
                       marker="^", label="Configured auspicious Tithi", zorder=4)
        ax.set(title=f"Chrysanthemum Prices on Festival and Auspicious-Tithi Days — {year}",
               xlabel="Date", ylabel="Price")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        money_axis(ax)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="best")
        save_plot(fig, f"event_auspicious_prices_{year}.png")

    # Event-day vs non-event-day price distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, box_data = [], []
    for label, mask in [
        ("Non-event day", ~data["Is_Event_Day"]),
        ("Festival / event day", data["Is_Event_Day"]),
    ]:
        values = data.loc[mask, PRICE_COL].dropna()
        if not values.empty:
            labels.append(label)
            box_data.append(values)
    ax.boxplot(box_data, labels=labels, showmeans=True)
    ax.set(title="Price Distribution: Event Days vs Non-Event Days",
           xlabel="Day category", ylabel="Price")
    money_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "11_event_vs_non_event_boxplot.png")

    # Tithi-wise average price
    tithi_avg = (data.dropna(subset=[TITHI_COL, PRICE_COL])
                 .groupby(TITHI_COL)[PRICE_COL].agg(["mean", "median", "count"])
                 .sort_values("mean", ascending=False))
    tithi_avg.to_csv(OUTPUT_DIR / "12_tithi_price_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 8))
    ordered = tithi_avg.sort_values("mean")
    ax.barh(ordered.index, ordered["mean"])
    ax.set(title="Average Chrysanthemum Price by Tithi",
           xlabel="Average price", ylabel="Tithi")
    money_axis(ax)
    ax.grid(axis="x", alpha=0.2)
    save_plot(fig, "13_average_price_by_tithi.png")

    # Month-by-year heatmap of monthly average prices (matplotlib only).
    pivot = monthly.pivot(index="Year", columns="Month", values="Average_Price").reindex(
        index=range(START_YEAR, END_YEAR + 1), columns=range(1, 13)
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set(title="Monthly Average Price Heatmap (2010–2025)",
           xlabel="Month", ylabel="Year")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Average price")
    save_plot(fig, "14_month_year_average_price_heatmap.png")

    # Yearly average trend
    annual = valid_dates.groupby("Year")[PRICE_COL].agg(["mean", "median", "min", "max", "std"])
    annual.to_csv(OUTPUT_DIR / "15_annual_price_statistics.csv")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(annual.index, annual["mean"], marker="o", label="Annual mean")
    ax.plot(annual.index, annual["median"], marker="s", label="Annual median")
    ax.set(title="Annual Chrysanthemum Price Trend (2010–2025)",
           xlabel="Year", ylabel="Price")
    ax.set_xticks(range(START_YEAR, END_YEAR + 1, 1))
    ax.tick_params(axis="x", rotation=45)
    money_axis(ax)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_plot(fig, "16_annual_mean_median_trend.png")

    # Event names with most records and average prices on event dates.
    event_rows = data[data["Is_Event_Day"]].copy()
    if not event_rows.empty:
        event_summary = (event_rows.groupby(EVENT_COL)[PRICE_COL]
                         .agg(["count", "mean", "median", "min", "max"])
                         .sort_values("count", ascending=False))
        event_summary.to_csv(OUTPUT_DIR / "17_event_name_price_summary.csv")

    print("\nEDA completed successfully.")
    print(f"Source CSV read only: {DATA_PATH}")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"Rows in source dataset: {len(raw):,}")
    print(f"Price min / max: {prices.min():,.2f} / {prices.max():,.2f}")
    print(f"Global IQR outliers flagged (not removed): {len(outliers):,}")
    print("Monthly average charts created for every year from 2010 through 2025.")
    print("Review AUSPICIOUS_TITHIS near the top and edit it to match your definition.")


if __name__ == "__main__":
    main()
