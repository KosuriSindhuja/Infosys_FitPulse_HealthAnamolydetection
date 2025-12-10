from pathlib import Path
import pandas as pd

# EDIT the filename below if needed — use the file shown in step 1
csv = Path("data/processed/preprocessed_data.csv")

print("Reading:", csv)
df = pd.read_csv(csv, low_memory=False)

print("\nShape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nMissing values (top 20):\n", df.isna().sum().sort_values(ascending=False).head(20))
print("\nSample rows:\n", df.head(8).to_string(index=False))