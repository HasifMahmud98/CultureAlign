# src/_step1_inspect_meta.py
import pandas as pd
import re
from pathlib import Path

# --- Anchor all paths to this script's location, not the CWD ---
SCRIPT_DIR = Path(__file__).resolve().parent        # .../CultureAlign/src
PROJECT_ROOT = SCRIPT_DIR.parent                    # .../CultureAlign
META = PROJECT_ROOT / "data" / "ShilpoBangla" / "Image_alt_text.xlsx"

print(f"Looking for: {META}")
print(f"Exists: {META.exists()}")
if not META.exists():
    # Show what's actually nearby, to help diagnose
    parent = META.parent
    if parent.exists():
        print(f"\nContents of {parent}:")
        for p in sorted(parent.iterdir()):
            print(f"  {p.name}")
    raise FileNotFoundError(f"Metadata file not found at {META}")

df = pd.read_excel(META)

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)

print("\nCOLUMNS:", df.columns.tolist())

print("\nDTYPES:")
print(df.dtypes.to_string())

print("\nNULL COUNTS:")
print(df.isna().sum().to_string())

print("\n" + "=" * 60)
print("FIRST 3 ROWS (full, not truncated):")
print("=" * 60)
for i in range(min(3, len(df))):
    print(f"\n--- Row {i} ---")
    for col in df.columns:
        print(f"  [{col}]: {df.iloc[i][col]}")

# --- Language detection heuristic ---
# Bangla Unicode block is U+0980–U+09FF
BANGLA_RE = re.compile(r"[\u0980-\u09FF]")

print("\n" + "=" * 60)
print("LANGUAGE CONTENT PER COLUMN (heuristic):")
print("=" * 60)
for col in df.columns:
    if df[col].dtype != object:
        print(f"  [{col}]: non-text column ({df[col].dtype})")
        continue
    sample = df[col].dropna().astype(str).head(200)
    has_bn = sample.apply(lambda s: bool(BANGLA_RE.search(s))).sum()
    has_latin = sample.apply(lambda s: bool(re.search(r"[A-Za-z]", s))).sum()
    print(f"  [{col}]: {has_bn}/{len(sample)} contain Bangla script, "
          f"{has_latin}/{len(sample)} contain Latin script")

# --- Category distribution ---
print("\n" + "=" * 60)
print("VALUE COUNTS FOR LOW-CARDINALITY TEXT COLUMNS:")
print("=" * 60)
for col in df.columns:
    if df[col].dtype == object:
        n_unique = df[col].nunique()
        if n_unique <= 20:
            print(f"\n[{col}] ({n_unique} unique):")
            print(df[col].value_counts().to_string())