from pathlib import Path
import pandas as pd

from config import DATA_DIR, CSV_PATH

def main():
    print(f"Project dataset folder: {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    csvs = [Path(CSV_PATH)] if CSV_PATH else sorted(DATA_DIR.rglob("*.csv"))
    if not csvs:
        print("No CSV files found. Extract the dataset under data/ShilpoBangla.")
        return

    print("\nCSV files found:")
    for i, path in enumerate(csvs, start=1):
        print(f"{i}. {path}")

    for path in csvs:
        print("\n" + "=" * 80)
        print("FILE:", path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print("Could not read CSV:", e)
            continue
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head().to_string(max_cols=20, max_colwidth=80))
        print("\nMissing values:")
        print(df.isna().sum().to_string())
        for col in ("class_name", "category", "class", "label"):
            if col in df.columns:
                print(f"\nCounts for {col}:")
                print(df[col].value_counts(dropna=False).to_string())
                break

if __name__ == "__main__":
    main()
