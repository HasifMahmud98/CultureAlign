# src/make_coding_sheet_v2.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"

# Load the previous sheet (already has: c1b_en, ref_alt_en, c2_bn, failure flags)
df = pd.read_csv(RES / "coding_sheet.csv", encoding="utf-8")

# Drop the old single-axis coding columns, add the new four-axis scheme
for old in ["code_attribution", "code_specificity", "code_notes"]:
    if old in df.columns:
        df = df.drop(columns=old)

df["ref_specificity"]  = ""   # specific / generic / encyclopedic / unclear
df["ref_attribution"]  = ""   # BD/IN/JP/AF/ME/CN/PK/SEA/generic/none
df["c1b_specificity"]  = ""   # specific / generic / encyclopedic / wrong
df["c1b_attribution"]  = ""   # BD/IN/JP/AF/ME/CN/PK/SEA/generic/multi/none
df["code_notes"]       = ""

# Reorder so the two text pairs sit next to their coding columns
order = [
    "row_id", "image_id", "category",
    "ref_alt_en", "ref_specificity", "ref_attribution",
    "c1b_en", "c1b_specificity", "c1b_attribution",
    "c2_bn", "script_class", "prompt_echo", "degenerate", "true_failure",
    "code_notes",
]
df = df[order]

out = RES / "coding_sheet_v2.csv"
df.to_csv(out, index=False, encoding="utf-8")
print(f"Wrote {len(df)} rows to {out}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 3 preview:")
print(df[["row_id","image_id","ref_alt_en","ref_specificity","ref_attribution",
          "c1b_en","c1b_specificity","c1b_attribution"]].head(3).to_string(index=False, max_colwidth=60))