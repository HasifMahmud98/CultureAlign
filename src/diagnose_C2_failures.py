# src/diagnose_C2_failures.py
import sys, re
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"

c1 = pd.read_csv(RES / "generated_C1b_en.csv", encoding="utf-8")
c2 = pd.read_csv(RES / "generated_C2_bn.csv", encoding="utf-8")

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE  = re.compile(r"[A-Za-z]")

def classify_script(t):
    if not isinstance(t, str): return "empty"
    bn = len(BANGLA_RE.findall(t)); la = len(LATIN_RE.findall(t))
    if bn == 0 and la == 0: return "empty"
    if bn > 0 and la == 0:  return "pure_bangla"
    if bn > 0 and la > 0:   return "mixed"
    return "pure_latin"

c2["script_class"] = c2["generated_text"].apply(classify_script)
c2["char_len"]     = c2["generated_text"].astype(str).str.len()

# --- Q1. Which categories fail? ---
print("=" * 70)
print("Q1. Failure rate by category")
print("=" * 70)
tab = pd.crosstab(c2["category"], c2["script_class"])
tab["total"]   = tab.sum(axis=1)
tab["fail_pct"] = ((tab.get("pure_latin", 0) + tab.get("mixed", 0)) / tab["total"] * 100).round(1)
print(tab.to_string())

# --- Q2. Refusals vs. collapse ---
print("\n" + "=" * 70)
print("Q2. Within the pure_latin rows: refusals vs. English collapse")
print("=" * 70)
latin = c2[c2.script_class == "pure_latin"].copy()

REFUSAL_PATTERNS = [
    r"sorry",
    r"can'?t assist",
    r"cannot assist",
    r"i'?m unable",
    r"i cannot provide",
    r"as an ai",
    r"inappropriate",
]
pattern = "|".join(REFUSAL_PATTERNS)
latin["is_refusal"] = latin["generated_text"].str.contains(pattern, case=False, regex=True, na=False)

n_refusals = latin.is_refusal.sum()
n_collapse = len(latin) - n_refusals
print(f"Total pure_latin rows: {len(latin)}")
print(f"  Refusals:        {n_refusals}")
print(f"  English collapse:{n_collapse}")

print("\n--- Refusal texts (all of them) ---")
for _, r in latin[latin.is_refusal].iterrows():
    print(f"  [{r.image_id}] ({r.category}): {r.generated_text[:160]!r}")

print("\n--- First 5 English-collapse texts ---")
for _, r in latin[~latin.is_refusal].head(5).iterrows():
    print(f"  [{r.image_id}] ({r.category}): {r.generated_text[:160]!r}")

# --- Q3. How is বাংলা being used? Ambiguous referent check ---
print("\n" + "=" * 70)
print("Q3. How is বাংলা used — language, region, or Bangladesh?")
print("=" * 70)
has_bangla_word = c2[c2.generated_text.astype(str).str.contains("বাংলা", regex=False)]
print(f"Rows containing 'বাংলা': {len(has_bangla_word)}")
print("\nFirst 12 occurrences with 40 chars of context:")
for _, r in has_bangla_word.head(12).iterrows():
    txt = str(r.generated_text)
    idx = txt.find("বাংলা")
    ctx = txt[max(0, idx-25):idx+35].replace("\n"," ")
    print(f"  [{r.image_id}]: …{ctx}…")

# --- Q4. Length asymmetry by category (Bangla vs English) ---
print("\n" + "=" * 70)
print("Q4. Median generation length by category (C1b vs C2), chars")
print("=" * 70)
c1["char_len"] = c1["generated_text"].astype(str).str.len()
c2_bangla = c2[c2.script_class == "pure_bangla"].copy()

g1 = c1.groupby("category")["char_len"].median().rename("C1b_en_chars")
g2 = c2_bangla.groupby("category")["char_len"].median().rename("C2_bn_chars")
print(pd.concat([g1, g2], axis=1).round(0).to_string())

# --- Save enriched sheet for the coding phase ---
latin["failure_type"] = latin["is_refusal"].map({True: "safety_refusal", False: "english_collapse"})
latin[["image_id", "category", "failure_type", "generated_text"]].to_csv(
    RES / "C2_failure_audit.csv", index=False, encoding="utf-8"
)
print(f"\nWrote results/C2_failure_audit.csv")