
import sys
 
import numpy as np
import pandas as pd
 
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "all_data_classified.csv"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "master_with_markers.csv"
WINDOW = 3  # days before / after
 
 
def marker(is_event, window=WINDOW):
    """-3..-1 = days before, 0 = event day, 1..3 = days after, NaN = outside window.
    If two events are close, the nearest one wins (ties -> the upcoming event)."""
    ev = np.flatnonzero(np.asarray(is_event, dtype=bool))
    out = np.full(len(is_event), np.nan)
    best = np.full(len(is_event), np.inf)
    for e in ev:
        for off in range(-window, window + 1):           # off = day - event_day
            i = e + off
            if 0 <= i < len(out):
                dist = abs(off)
                if dist < best[i] or (dist == best[i] and off < 0):
                    best[i], out[i] = dist, off
    return out
 
 
def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["festival_marker"] = marker(df["is_festival"])
    df["auspicious_marker"] = marker(df["is_auspicious"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()
 


