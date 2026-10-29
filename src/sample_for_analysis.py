# src/sample_for_analysis.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META      = PROJECT_ROOT / "data" / "ShilpoBangla" / "Image_alt_text.xlsx"
DATA_ROOT = PROJECT_ROOT / "data" / "ShilpoBangla" / "ShilpoBangla"
OUT       = PROJECT_ROOT / "results" / "sample_manifest.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(META)

df = df.rename(columns={
    "class_name": "category",
    "image_path": "rel_image_path",
    "alt_text_bn": "ref_alt_bn",
    "eng_text":    "ref_alt_en",
})

df["image_path"] = df["rel_image_path"].apply(lambda p: str(DATA_ROOT / p))
df["image_id"]   = df["rel_image_path"].apply(lambda p: Path(p).stem)
df["item_cluster"] = (
    df["ref_alt_en"].astype(str).str.slice(0, 40).str.lower().str.strip()
)

missing = [p for p in df["image_path"] if not Path(p).exists()]
print(f"Missing files: {len(missing)} / {len(df)}")
if missing:
    print("First 5 missing:", missing[:5])

# FIXED: use groupby().sample() — preserves all columns including 'category'
N_PER_CAT = 50
sampled = (
    df.groupby("category")
      .sample(n=N_PER_CAT, random_state=42)
      .reset_index(drop=True)
)

print(f"\nSampled columns: {sampled.columns.tolist()}")
sampled.to_csv(OUT, index=False)
print(f"Wrote {len(sampled)} rows to {OUT}")

print("\nCategory distribution:")
print(sampled["category"].value_counts().to_string())

print("\nDiversity metrics:")
print(f"  Unique item clusters in FULL dataset: {df['item_cluster'].nunique():>4} / {len(df)}")
print(f"  Unique item clusters in SAMPLE:       {sampled['item_cluster'].nunique():>4} / {len(sampled)}")
print(f"  Unique Bangla references in sample:   {sampled['ref_alt_bn'].nunique():>4}")
print(f"  Unique English references in sample:  {sampled['ref_alt_en'].nunique():>4}")

print("\nFirst 5 rows of sample:")
print(sampled[["image_id", "category", "ref_alt_en"]].head(5).to_string(index=False))