import pandas as pd
from pathlib import Path
import os

input_path = Path(r"C:\Users\ASUS\Downloads\DDW_PCA0000_2011_Indiastatedist.xlsx")
output_path = Path(r"F:\Water Scarcity Prediction\DATA FILES\india_district_population_2011.csv")

# check input exists
if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

# load excel (ensure openpyxl is installed)
df = pd.read_excel(input_path)

# filter district level
district_df = df[df["Level"] == "DISTRICT"].copy()

# keep relevant columns
cols = ["State", "District", "Name", "TRU", "TOT_P", "TOT_M", "TOT_F"]
missing = [c for c in cols if c not in district_df.columns]
if missing:
    raise KeyError(f"Missing expected columns: {missing}")

district_df = district_df[cols]

# keep only Total rows (not Rural/Urban)
district_df = district_df[district_df["TRU"].str.strip().str.lower() == "total"]

# drop TRU and rename
district_df = district_df.drop(columns=["TRU"]).rename(columns={
    "Name": "District_Name",
    "TOT_P": "Total_Population",
    "TOT_M": "Male_Population",
    "TOT_F": "Female_Population"
}).reset_index(drop=True)

# ensure output folder exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# save csv
district_df.to_csv(output_path, index=False)
print("Saved cleaned CSV to:", output_path)
