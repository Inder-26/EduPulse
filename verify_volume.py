
import pandas as pd
import glob
import os

data_dir = r"d:\Coding Central\Business_Optima_Assignment\data"
all_files = glob.glob(os.path.join(data_dir, "*.csv"))

print(f"Checking {len(all_files)} files in {data_dir}...\n")

total_rows = 0
individual_files_rows = 0
merged_file_rows = 0

for f in all_files:
    try:
        df = pd.read_csv(f)
        fname = os.path.basename(f)
        print(f"File: {fname} | Rows: {len(df)} | Cols: {list(df.columns)}")
        
        if fname == "all_reviews_merged.csv":
            merged_file_rows = len(df)
        else:
            individual_files_rows += len(df)
            
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\n--- Summary ---")
print(f"Total rows in individual coursework files: {individual_files_rows}")
print(f"Total rows in 'all_reviews_merged.csv': {merged_file_rows}")

if abs(individual_files_rows - merged_file_rows) < 100:
    print("\n✅ Data Verification: The 'merged' file matches the individual files.")
else:
    print("\n⚠️ Data Mismatch: There is a significant difference in row counts.")
