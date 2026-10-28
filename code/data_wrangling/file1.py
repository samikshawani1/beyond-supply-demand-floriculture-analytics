# to combine the data in flower prices different worksheets into a single csv file.

import pandas as pd

# Read all worksheets from the XLS file
file = pd.ExcelFile(r"C:\Users\Ghanavi Arun\Floriculture\data\Flower price 2019-25.xls")

# Combine all worksheets into one DataFrame
data = []

for sheet in file.sheet_names:
    df = pd.read_excel(file, sheet_name=sheet)
    df["Source_Sheet"] = sheet  # Optional: track worksheet name
    data.append(df)

# Combine all worksheets vertically
combined = pd.concat(data, ignore_index=True, sort=False)

# Save everything into one CSV file
combined.to_csv("output2.csv", index=False)

print("All worksheets combined into output.csv")