#concatenate output1 and output2 to flower_prices.csv

import pandas as pd

# Read both CSV files
df1 = pd.read_csv(r"C:\Users\Ghanavi Arun\Floriculture\output1.csv")
df2 = pd.read_csv(r"C:\Users\Ghanavi Arun\Floriculture\output2.csv")

# Append csv2 rows to csv1
combined = pd.concat([df1, df2], ignore_index=True)

# Save the result back to csv1
combined.to_csv("flower_prices.csv", index=False)

print("CSV2 data appended to CSV1 successfully!")