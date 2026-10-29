# src/make_coding_sheet.py
import pandas as pd
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"
META = PROJECT_ROOT / "data" / "ShilpoBangla" / "Image_alt_text.xlsx"

# --- Load ---
c1 = pd.read_csv(RES / "generated_C1b_en.csv", encoding="utf-8")
c2 = pd.read_csv(RES / "C2_enriched.csv", encoding="utf-8")
meta = pd.read_excel(META).rename(columns={
    "image_path": "rel_image_path",
    "alt_text_bn": "ref_alt_bn",
    "eng_text":    "ref_alt_en",
    "class_name":  "category",
})
meta["image_id"] = meta["rel_image_path"].apply(lambda p: Path(p).stem)

# --- Recompute 'degenerate' on the full 250 rows ---
# (Same logic as audit_degeneration.py; must be here because
#  C2_enriched.csv predates that column being added.)
TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z]+")

def _max_ngram_repetition(text, n=3):
    if not isinstance(text, str): return 0
    toks = TOKEN_RE.findall(text.lower())
    if len(toks) < n: return 0
    grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    if not grams: return 0
    return Counter(grams).most_common(1)[0][1]

def is_degenerate(text):
    if not isinstance(text, str) or len(text) < 40:
        return False
    if _max_ngram_repetition(text, n=3) >= 3:
        return True
    words = TOKEN_RE.findall(text.lower())
    if len(words) >= 8:
        top_word, top_cnt = Counter(words).most_common(1)[0]
        if top_cnt >= 4 and top_cnt / len(words) > 0.25:
            return True
    return False

c2["degenerate"] = c2["generated_text"].apply(is_degenerate)

# --- Merge ---
df = c1[["image_id", "category", "generated_text"]].rename(
    columns={"generated_text": "c1b_en"}
)
df = df.merge(
    c2[["image_id", "script_class", "prompt_echo", "degenerate",
        "true_failure", "generated_text"]].rename(
            columns={"generated_text": "c2_bn"}
        ),
    on="image_id", how="left",
)
df = df.merge(
    meta[["image_id", "ref_alt_en", "ref_alt_bn"]],
    on="image_id", how="left",
)

# --- Coding columns (empty, for you to fill) ---
df["code_attribution"] = ""   # BD/IN/JP/AF/ME/CN/PK/SEA/generic/multi/self/other
df["code_specificity"] = ""   # specific_object/generic_type/category_only/wrong_object
df["code_notes"]       = ""

# --- Shuffle so categories don't cluster; reset row numbers ---
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "row_id", range(1, len(df) + 1))

# --- Save ---
out = RES / "coding_sheet.csv"
df.to_csv(out, index=False, encoding="utf-8")

# --- Save augmented C2 back so future scripts don't hit the same bug ---
c2.to_csv(RES / "C2_enriched.csv", index=False, encoding="utf-8")

# --- Report ---
print(f"Wrote {len(df)} rows to {out}")
print(f"Updated {RES/'C2_enriched.csv'} (added 'degenerate' column)")
print("\nColumns:", df.columns.tolist())
print(f"\nDegenerate rows in full 250: {df.degenerate.sum()}")
print(f"Prompt echoes:               {df.prompt_echo.sum()}")
print(f"True failures (older flag):  {df.true_failure.sum()}")
print(f"\nFirst 3 rows (preview):")
print(df[["row_id", "image_id", "category", "c1b_en", "ref_alt_en"]]
      .head(3).to_string(index=False))