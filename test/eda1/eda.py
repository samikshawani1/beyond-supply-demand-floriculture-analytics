from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\master_with_markers.csv"
OUT = Path(__file__).parent / "eda_output"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
df["Year"] = df["Date"].dt.year

# ---------------------------------------------------------------- 1. STRUCTURE
print("=" * 60, "\n1. DATA STRUCTURE\n", "=" * 60)
print("Rows, columns :", df.shape)
print("Date range    :", df["Date"].min().date(), "to", df["Date"].max().date())
all_days = pd.date_range(df["Date"].min(), df["Date"].max())
print("Missing dates :", len(all_days.difference(df["Date"])))
print("Duplicate dates:", df["Date"].duplicated().sum())
print("\nColumn types and missing values:")
info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                     "unique": df.nunique()})
print(info)
print("\nFirst 5 rows:\n", df.head())
print("\nPrice summary:\n", df["Price"].describe().round(1))
for col in ["festival_marker", "auspicious_marker"]:
    if col in df.columns:
        print(f"\n{col} value counts (blank = outside window):")
        print(df[col].value_counts(dropna=False).sort_index())
info.to_csv(OUT / "data_structure.csv")

# ---------------------------------------------------------------- 2. OUTLIERS
# Method: compare each price with the median of the surrounding 31 days.
# Flag a day if it is more than 3x higher or lower than that local median.
df["local_median"] = df["Price"].rolling(31, center=True, min_periods=15).median()
df["ratio"] = df["Price"] / df["local_median"]
df["outlier"] = (df["ratio"] > 3) | (df["ratio"] < 1 / 3)
out = df[df["outlier"]][["Date", "Price", "local_median", "ratio", "Events"]]
print("\n" + "=" * 60, "\n2. OUTLIERS (price > 3x or < 1/3 of its local median)\n", "=" * 60)
print("Number of outliers:", len(out))
print("Outliers per year:\n", out.groupby(out["Date"].dt.year).size())
print("\nTop 10 by ratio:\n", out.assign(r=np.log(out["ratio"]).abs())
      .sort_values("r", ascending=False).head(10).drop(columns="r").to_string(index=False))
out.to_csv(OUT / "outliers.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 3. YEARLY TRENDS
yearly = df.groupby("Year")["Price"].agg(["mean", "median", "min", "max", "std"]).round(1)
yearly["change_in_median_%"] = (yearly["median"].pct_change() * 100).round(1)
print("\n" + "=" * 60, "\n3. YEARLY PRICE SUMMARY\n", "=" * 60)
print(yearly)
yearly.to_csv(OUT / "yearly_summary.csv")

# ---------------------------------------------------------------- 4. GRAPHS
# Graph 1: full history with outliers in red
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df["Date"], df["Price"], lw=0.5, color="steelblue", label="Daily price")
ax.scatter(out["Date"], out["Price"], color="red", s=15, zorder=3, label="Outlier")
ax.set(title="Daily price 2010-2025 (outliers in red)", xlabel="Date", ylabel="Price")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "1_price_history_outliers.png", dpi=120); plt.close()

# Graph 2: yearly mean / median / min / max
fig, ax = plt.subplots(figsize=(11, 5))
for c in ["mean", "median", "min", "max"]:
    ax.plot(yearly.index, yearly[c], marker="o", label=c)
ax.set(title="Yearly price trend", xlabel="Year", ylabel="Price")
ax.set_xticks(yearly.index); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "2_yearly_trend.png", dpi=120); plt.close()

# Graph 3: boxplot per year (dots outside the box = unusual prices)
fig, ax = plt.subplots(figsize=(13, 5))
ax.boxplot([g["Price"] for _, g in df.groupby("Year")])
ax.set_xticklabels(yearly.index)
ax.set(title="Price spread by year", xlabel="Year", ylabel="Price")
fig.tight_layout(); fig.savefig(OUT / "3_boxplot_by_year.png", dpi=120); plt.close()

# Graph 4: one small chart per year (daily price + 30-day average)
fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharey=True)
for ax, (year, g) in zip(axes.ravel(), df.groupby("Year")):
    ax.plot(g["Date"], g["Price"], lw=0.5, color="lightsteelblue")
    ax.plot(g["Date"], g["Price"].rolling(30, min_periods=10).mean(), lw=1.5, color="navy")
    o = g[g["outlier"]]
    ax.scatter(o["Date"], o["Price"], color="red", s=10, zorder=3)
    ax.set_title(str(year)); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
fig.suptitle("Price trend inside each year (dark line = 30-day average, red = outlier)")
fig.tight_layout(); fig.savefig(OUT / "4_each_year.png", dpi=120); plt.close()

# Graph 5: price distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["Price"], bins=60, color="steelblue"); ax[0].set_title("Price distribution")
ax[1].hist(np.log(df["Price"]), bins=60, color="seagreen"); ax[1].set_title("log(Price) distribution")
fig.tight_layout(); fig.savefig(OUT / "5_distribution.png", dpi=120); plt.close()

print("\nAll tables and graphs saved in:", OUT)