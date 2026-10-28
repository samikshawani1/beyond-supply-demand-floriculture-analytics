import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = r"C:\Users\Ghanavi Arun\Floriculture\data\bronze\master_with_markers.csv"
OUT = Path(__file__).parent / "eda_output"
YEARS = None            # None = all years, or e.g. [2023] / [2022, 2023]

# festivals highlighted in dark red (provisional list - edit as you like)
MAJOR = ["Ugadi", "Makara Sankranti", "Maha Shivaratri", "Ganesh Chaturthi", "Janmashtami",
         "Varalakshmi Vrat", "Navratri Begins", "Dussehra", "Diwali", "Tulasi Vivah",
         "Rama Navami", "Akshaya Tritiya", "Nag Panchami", "Raksha Bandhan", "Onam"]
MAJOR_RE = "|".join(r"(?<!Ganga )Dussehra" if m == "Dussehra" else re.escape(m) for m in MAJOR)

(OUT / "festivals_by_year").mkdir(parents=True, exist_ok=True)
(OUT / "auspicious_by_year").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
df["Events"] = df["Events"].fillna("")
df["Year"] = df["Date"].dt.year
df["is_major"] = df["Events"].str.contains(MAJOR_RE)
df["avg30"] = df["Price"].rolling(30, min_periods=10).mean()
years = YEARS or sorted(df["Year"].unique())


def short_name(text):
    """'Surya Grahan *Anshika | Holi' -> 'Surya Grahan / Holi' (max ~34 chars)"""
    names = [re.split(r"\s[*@]", a.strip())[0] for a in text.split("|")]
    s = " / ".join(names)
    return s if len(s) <= 30 else s[:28] + ".."


def style_months(ax, year):
    ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31) + pd.Timedelta(days=14))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def price_ylim(ax, g):
    ax.set_ylim(0, np.percentile(g["Price"], 99.5) * 1.15)   # keeps one-off spikes from flattening the chart


# ------------------------------------------------------------ one chart per year: FESTIVALS
for year in years:
    g = df[df["Year"] == year]
    f = g[g["is_festival"]].copy()

    # pack the name labels into rows so that no two labels on the same row overlap
    f["label"] = f["Events"].apply(short_name)
    row_end, rows = [], []
    for d, text in zip(f["Date"], f["label"]):
        start = d.toordinal()
        width = len(text) * 1.3 + 3                       # approx. width of the text in days
        r = next((i for i, e in enumerate(row_end) if e <= start), len(row_end))
        if r == len(row_end):
            row_end.append(0)
        row_end[r] = start + width
        rows.append(r)
    f["row"] = rows
    n_rows = max(rows) + 1

    fig = plt.figure(figsize=(19, 6 + 0.3 * n_rows))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 0.3 * n_rows], hspace=0.12)
    ax, lab = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.plot(g["Date"], g["Price"], lw=0.6, color="lightsteelblue", label="Daily price")
    ax.plot(g["Date"], g["avg30"], lw=1.6, color="navy", label="30-day average")
    for _, r in f.iterrows():
        ax.axvline(r["Date"], color="darkred" if r["is_major"] else "red",
                   alpha=0.35 if r["is_major"] else 0.12, lw=1.2 if r["is_major"] else 0.8)
    ax.scatter(f["Date"], f["Price"], color="red", s=16, zorder=3, label="Festival day")
    price_ylim(ax, g); style_months(ax, year); ax.set_ylabel("Price"); ax.legend(loc="upper left")
    ax.set_title(f"{year}: price and all {len(f)} festival days (dark red = major festivals)")

    for _, r in f.iterrows():
        y = n_rows - 1 - r["row"]
        lab.vlines(r["Date"], y, n_rows, color="grey", lw=0.5)
        lab.text(r["Date"], y, f" {r['label']}", fontsize=6.8, va="center", ha="left",
                 color="darkred" if r["is_major"] else "black",
                 fontweight="bold" if r["is_major"] else "normal")
    style_months(lab, year); lab.set_ylim(-0.6, n_rows); lab.set_yticks([])
    lab.grid(axis="x", alpha=0.3); lab.tick_params(labelbottom=False)
    for sp in ["top", "right", "left"]:
        lab.spines[sp].set_visible(False)
    fig.savefig(OUT / "festivals_by_year" / f"festivals_{year}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------ one chart per year: AUSPICIOUS DAYS
def day_type(row):
    t = row["Events"]
    if row["is_festival"]:
        return "Festival"
    for key, name in [("Ekadashi", "Ekadashi"), ("Purnima", "Purnima"), ("Amavas", "Amavasya"),
                      ("Sankranti", "Sankranti")]:
        if key in t:
            return name
    return "Other"


COLORS = {"Festival": "red", "Ekadashi": "green", "Purnima": "orange", "Amavasya": "purple",
          "Sankranti": "teal", "Other": "grey"}
df["aus_type"] = ""
df.loc[df["is_auspicious"], "aus_type"] = df[df["is_auspicious"]].apply(day_type, axis=1)

for year in years:
    g = df[df["Year"] == year]
    a = g[g["is_auspicious"]]
    other = g[~g["is_auspicious"]]
    fig, (ax, st) = plt.subplots(2, 1, figsize=(17, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.plot(g["Date"], g["Price"], lw=0.6, color="lightsteelblue", label="Daily price")
    ax.plot(g["Date"], g["avg30"], lw=1.6, color="navy", label="30-day average")
    ax.scatter(a["Date"], a["Price"], color="green", s=14, zorder=3, label="Auspicious day")
    price_ylim(ax, g); style_months(ax, year); ax.set_ylabel("Price"); ax.legend(loc="upper left")
    ax.set_title(f"{year}: {len(a)} auspicious days | mean price on auspicious days "
                 f"{a['Price'].mean():.0f} vs other days {other['Price'].mean():.0f}")
    for name, col in COLORS.items():
        d = a[a["aus_type"] == name]["Date"]
        st.vlines(d, 0, 1, color=col, lw=2, label=f"{name} ({len(d)})")
    st.set_yticks([]); st.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.savefig(OUT / "auspicious_by_year" / f"auspicious_{year}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------ overview of all years
for kind, title in [("festival", "Festival days (dark red = major)"), ("auspicious", "Auspicious days")]:
    fig, axes = plt.subplots(4, 4, figsize=(18, 11), sharey=True)
    for ax, year in zip(axes.ravel(), years):
        g = df[df["Year"] == year]
        ax.plot(g["Date"], g["avg30"], color="navy", lw=1.2)
        if kind == "festival":
            for _, r in g[g["is_festival"]].iterrows():
                ax.axvline(r["Date"], color="darkred" if r["is_major"] else "red",
                           alpha=0.5 if r["is_major"] else 0.12, lw=1 if r["is_major"] else 0.6)
        else:
            a = g[g["is_auspicious"]]
            ax.scatter(a["Date"], a["avg30"], color="green", s=4, alpha=0.5)
        ax.set_title(str(year)); style_months(ax, year)
        ax.set_xticks(pd.date_range(f"{year}-01-01", periods=4, freq="3MS"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle(f"30-day average price with {title}, all years")
    fig.tight_layout(); fig.savefig(OUT / f"overview_{kind}_all_years.png", dpi=110); plt.close(fig)

print("Saved to:", OUT)