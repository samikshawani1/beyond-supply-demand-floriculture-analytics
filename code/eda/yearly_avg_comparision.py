# file: C:\Users\Ghanavi Arun\Floriculture\code\eda\yearly_avg_comparision.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 1. Paths & Setup
# ---------------------------------------------------------------------------
INPUT = Path(r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\all_data_classified.csv")
OUT   = Path(r"C:\Users\Ghanavi Arun\Floriculture\code\eda")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load & Prep Data
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT, encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Year"] = df["Date"].dt.year

# Drop rows missing essential info
df = df.dropna(subset=["Year", "Price"]).copy()
df["Year"] = df["Year"].astype(int)

# Identify Festival/Event Days
df["Events"] = df["Events"].fillna("").astype(str).str.strip()
df["is_festival"] = df["is_festival"].astype(str).str.lower().eq("true")
df["has_event"] = df["Events"].ne("")

# A day is "Festival/Event" if it's a festival OR has any event listed
df["Day Type"] = np.where(df["is_festival"] | df["has_event"], 
                          "Festival/Event Days", 
                          "Ordinary Days")

# ---------------------------------------------------------------------------
# 3. Calculate Yearly Averages
# ---------------------------------------------------------------------------
# Group by Year and Day Type, then calculate the mean price
yearly_avg = df.groupby(["Year", "Day Type"])["Price"].mean().unstack()

# Ensure both columns exist (in case a year has no festival or no ordinary days)
yearly_avg = yearly_avg.reindex(columns=["Ordinary Days", "Festival/Event Days"])
yearly_avg = yearly_avg.fillna(0)

# Save the summary to CSV just in case you need the numbers
yearly_avg.to_csv(OUT / "yearly_festival_event_vs_ordinary.csv")

# ---------------------------------------------------------------------------
# 4. Plotting the Grouped Bar Chart
# ---------------------------------------------------------------------------
years = yearly_avg.index
ordinary_prices = yearly_avg["Ordinary Days"]
festival_prices = yearly_avg["Festival/Event Days"]

x = np.arange(len(years))  # label locations
width = 0.35               # width of the bars

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Colors matching your reference image (Muted Blue & Muted Red)
color_ordinary = "#4C72B0" 
color_festival = "#C44E52"

# Plot bars
bars1 = ax.bar(x - width/2, ordinary_prices, width, label="Ordinary Days", color=color_ordinary)
bars2 = ax.bar(x + width/2, festival_prices, width, label="Festival/Event Days", color=color_festival)

# ---------------------------------------------------------------------------
# 5. Styling the Chart
# ---------------------------------------------------------------------------
# Add data labels on top of the bars (e.g., ₹236)
ax.bar_label(bars1, fmt='₹%.0f', padding=3, fontsize=9, color='#222222')
ax.bar_label(bars2, fmt='₹%.0f', padding=3, fontsize=9, color='#222222')

# Formatting axes
ax.set_ylabel("Average Price (₹)", fontsize=12, fontweight='bold')
ax.set_xlabel("Year", fontsize=12, fontweight='bold')
ax.set_title("Average Flower Price by Year (2010–2025)", fontsize=14, fontweight='bold', pad=15)

# Set x-ticks
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=11)

# Add horizontal grid lines behind the bars
ax.grid(axis='y', linestyle='-', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Add legend
ax.legend(title="Day Type", title_fontsize='11', fontsize='11', loc='upper left')

# Set Y-axis limits to give some headroom for the labels
max_price = max(ordinary_prices.max(), festival_prices.max())
ax.set_ylim(0, max_price * 1.15)

# Add a figure caption at the bottom
plt.figtext(0.5, 0.01, "Figure: Average flower price by year (Festival/Event vs Ordinary)", 
            ha="center", fontsize=12)

# Adjust layout so nothing gets cut off
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save and show
plot_path = OUT / "avg_price_by_year_grouped_bar.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Chart saved to: {plot_path}")

plt.show()