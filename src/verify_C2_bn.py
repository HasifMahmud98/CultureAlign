# src/verify_C2_bn.py
import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"

c1 = pd.read_csv(RES / "generated_C1b_en.csv")
c2 = pd.read_csv(RES / "generated_C2_bn.csv")

# ---------- 1. SCRIPT STATISTICS ----------
BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE  = re.compile(r"[A-Za-z]")

def script_profile(text):
    if not isinstance(text, str):
        return "empty", 0, 0
    bn = len(BANGLA_RE.findall(text))
    la = len(LATIN_RE.findall(text))
    if bn == 0 and la == 0:    return "empty", 0, 0
    if bn > 0 and la == 0:     return "pure_bangla", bn, la
    if bn > 0 and la > 0:      return "mixed", bn, la
    return "pure_latin", bn, la

c2[["script_class", "n_bn", "n_latin"]] = c2["generated_text"].apply(
    lambda t: pd.Series(script_profile(t))
)

print("=" * 70)
print("SECTION 1 — SCRIPT STATISTICS (C2 Bangla generations)")
print("=" * 70)
print(f"Total rows: {len(c2)}")
print(f"\nScript class distribution:")
print(c2["script_class"].value_counts().to_string())

# Flag rows that are not pure_bangla
not_pure = c2[c2["script_class"] != "pure_bangla"]
if len(not_pure) > 0:
    print(f"\nNon-pure-Bangla rows ({len(not_pure)}):")
    for _, r in not_pure.head(10).iterrows():
        preview = str(r.generated_text)[:100]
        print(f"  [{r.image_id}] ({r.script_class}) bn_chars={r.n_bn} latin_chars={r.n_latin}")
        print(f"    {preview}")
else:
    print("\n✓ All C2 rows are pure Bangla script.")

# Character-length distribution (proxy for content richness)
c2["char_len"] = c2["generated_text"].astype(str).str.len()
print(f"\nCharacter-length stats (Bangla text):")
print(f"  min={c2.char_len.min()}  median={c2.char_len.median():.0f}  "
      f"mean={c2.char_len.mean():.0f}  max={c2.char_len.max()}")
short = c2[c2.char_len < 30]
if len(short) > 0:
    print(f"\n  Rows under 30 chars ({len(short)}) — likely truncated/failed:")
    for _, r in short.head(5).iterrows():
        print(f"    [{r.image_id}]: {r.generated_text!r}")

# ---------- 2. CULTURAL ATTRIBUTION COUNT (Bangla terms) ----------
print("\n" + "=" * 70)
print("SECTION 2 — CULTURAL ATTRIBUTION KEYWORDS")
print("=" * 70)

# Bangla keywords for major referents
BN_KEYWORDS = {
    "Bangladesh (বাংলাদেশ)": "বাংলাদেশ",
    "Bengal/Bangla (বাংলা)": "বাংলা",
    "Bengali (বাঙালি)":       "বাঙালি",
    "India (ভারত)":           "ভারত",
    "Indian (ভারতীয়)":       "ভারতীয়",
    "Japan/Japanese (জাপান)": "জাপান",
    "Africa/African (আফ্রিকা)": "আফ্রিকা",
    "Middle East (মধ্যপ্রাচ্য)": "মধ্যপ্রাচ্য",
    "China/Chinese (চীন)":    "চীন",
    "Pakistani (পাকিস্তান)":  "পাকিস্তান",
    "Nepal (নেপাল)":          "নেপাল",
}

print(f"{'Term':30s} {'C2 (bn)':>10s} {'C1b (en)':>10s}")
for label, kw in BN_KEYWORDS.items():
    n_bn = c2.generated_text.astype(str).str.contains(kw, regex=False).sum()
    # For English, use the English word
    en_kw = label.split("(")[0].strip().rstrip("/").split("/")[0]
    # simpler: strip the bangla part
    en_term = label.split("(")[0].strip()
    if "/" in en_term:
        en_terms = [t.strip() for t in en_term.split("/")]
    else:
        en_terms = [en_term]
    n_en = sum(c1.generated_text.astype(str).str.contains(t, case=False, regex=False).sum()
               for t in en_terms)
    print(f"{label:30s} {n_bn:>10d} {n_en:>10d}")

# Specific comparison: Bangladesh vs India in each condition
print("\n" + "-" * 70)
print("Headline comparison:")
print("-" * 70)
bd_bn = c2.generated_text.astype(str).str.contains("বাংলাদেশ", regex=False).sum()
in_bn = c2.generated_text.astype(str).str.contains("ভারত",     regex=False).sum()
bd_en = c1.generated_text.astype(str).str.contains("Bangladesh", case=False, regex=False).sum()
in_en = c1.generated_text.astype(str).str.contains("India",      case=False, regex=False).sum()
print(f"  C1b (English):  Bangladesh mentioned {bd_en:>3d} / {len(c1)}   "
      f"India mentioned {in_en:>3d} / {len(c1)}")
print(f"  C2  (Bangla):   বাংলাদেশ mentioned {bd_bn:>3d} / {len(c2)}   "
      f"ভারত mentioned {in_bn:>3d} / {len(c2)}")

# ---------- 3. SIDE-BY-SIDE SAMPLE ----------
print("\n" + "=" * 70)
print("SECTION 3 — SIDE-BY-SIDE SAMPLE (C1b vs C2, same image)")
print("=" * 70)

merged = c1.merge(c2, on="image_id", suffixes=("_en", "_bn"))
sample = merged.head(10)
for _, r in sample.iterrows():
    print(f"\n[{r.image_id}]  ({r.category_en})")
    print(f"  REF_EN : {str(r.ref_alt_en if 'ref_alt_en' in r else '')[:200]}")  # may not be present
    print(f"  C1b_EN : {str(r.generated_text_en)[:200]}")
    print(f"  C2_BN  : {str(r.generated_text_bn)[:200]}")

# ---------- 4. WRITE A SUMMARY CSV FOR FURTHER INSPECTION ----------
summary = c2[["image_id", "category", "script_class", "n_bn", "char_len", "generated_text"]].copy()
summary["c1b_en"] = summary["image_id"].map(
    c1.set_index("image_id")["generated_text"]
)
summary.to_csv(RES / "C2_verification_summary.csv", index=False)
print(f"\nWrote detailed summary to: results/C2_verification_summary.csv")
print("Open this CSV to scroll through all 250 pairs side-by-side.")