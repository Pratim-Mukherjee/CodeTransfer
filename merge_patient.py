import pandas as pd
import glob
import os

data_folder = "archive (1)/data"
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

all_data = []
for file in csv_files:
    try:
        df = pd.read_csv(file, header=None)          # no header in individual files
        patient_id = os.path.basename(file).replace('.csv', '')
        df.insert(0, 'patient_id', patient_id)
        all_data.append(df)
    except Exception as e:
        print(f"Error with {file}: {e}")

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('all_patients.csv', index=False)
    print(f"✅ Merged {len(csv_files)} files → all_patients.csv ({len(combined_df)} rows)")
else:
    print("❌ No valid CSV files found!")