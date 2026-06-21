import pandas as pd
from pathlib import Path

Data_direction = Path("Data")
OUTPUT_FILE = Path("combined_data.csv")

#load the data once
dfs = []

for folder in Data_direction.iterdir():
    if folder.is_dir():
        for file in folder.glob("*.csv"):
            df = pd.read_csv(file)
            dfs.append(df)

# combine all dfs and create one csv
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(OUTPUT_FILE, index=False)
