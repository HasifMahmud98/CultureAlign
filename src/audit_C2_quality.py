# src/audit_C2_quality.py
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
audit = pd.read_csv(RES / "C2_failure_audit.csv", encoding="utf-8")

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

# --- Prompt echo detection ---
PROMPT_PHRASES = [
    "সংক্ষিপ্ত, তথ্যমূলক বর্ণনা বাংলায় লেখো",
    "ভূমিকা যোগ করো না",
    "অনিশ্চিত হলে সেই বিবরণ বাদ দাও",
    "যতটা নিশ্চিত ততটাই লেখো",
    "সাংস্কৃতিক বস্তুটির জন্য",
]
def has_prompt_echo(t):
    return isinstance(t, str) and any(p in t for p in PROMPT_PHRASES)

# --- Meta-response detection ---
META_PHRASES = [
    "please provide more information",
    "leave the information blank",
    "this is a brief",
    "this is a short",
    "it is not certain",
    "based on the image",
    "as an ai",
    "i cannot provide",
]
def is_meta(t):
    return isinstance(t, str) and any(p in t.lower() for p in META_PHRASES)

c2["prompt_echo"]   = c2["generated_text"].apply(has_prompt_echo)
c2["meta_response"] = c2["generated_text"].apply(is_meta)

print("=" * 70)
print("D1. PROMPT ECHOES AND META-RESPONSES")
print("=" * 70)
print(f"Total C2 rows: {len(c2)}")
print(f"  Prompt echo detected: {c2.prompt_echo.sum()}")
print(f"  Meta response detected: {c2.meta_response.sum()}")
echo_bn = c2[c2.prompt_echo & (c2.script_class == "pure_bangla")]
print(f"  Pure_bangla + prompt echo (NOT real descriptions): {len(echo_bn)}")

if len(echo_bn) > 0:
    print("\nSample prompt echoes:")
    for _, r in echo_bn.head(5).iterrows():
        print(f"  [{r.image_id}]: {r.generated_text[:180]}")

# --- Tautology detection in pure_bangla ---
pure_bn = c2[c2.script_class == "pure_bangla"].copy()
def is_tautology(t):
    if not isinstance(t, str): return False
    return t.count("সাংস্কৃতিক বস্তু") >= 2

pure_bn["tautology"] = pure_bn["generated_text"].apply(is_tautology)

print("\n" + "=" * 70)
print("D2. QUALITY OF THE 173 PURE-BANGLA ROWS")
print("=" * 70)
print(f"Total pure_bangla: {len(pure_bn)}")
print(f"  Prompt echo:         {pure_bn.prompt_echo.sum() if 'prompt_echo' in pure_bn else 'n/a'}")
print(f"  Tautological:        {pure_bn.tautology.sum()}")

# Valid Bangla = pure_bangla, not echo, not tautology
valid_bn = pure_bn[(~pure_bn.prompt_echo) & (~pure_bn.tautology)]
valid_pct = len(valid_bn) / 250 * 100
print(f"\nValid Bangla descriptions (not echo, not tautology): {len(valid_bn)}/250 ({valid_pct:.1f}%)")

# --- Refusal gender correlation ---
print("\n" + "=" * 70)
print("D3. REFUSALS: gender-coded content in the C1b English description?")
print("=" * 70)
refused_ids = audit[audit.failure_type == "safety_refusal"].image_id.tolist()
GENDER_TERMS = ["woman", "women", "female", "girl", "lady", "she ", "her ", "worn by"]
print(f"Total refusals in C2: {len(refused_ids)}")
print()
for iid in refused_ids:
    row = c1[c1.image_id == iid]
    if len(row) == 0:
        print(f"  [{iid}] NOT in C1b file")
        continue
    c1_text = str(row.generated_text.iloc[0])
    hits = [t for t in GENDER_TERMS if t in c1_text.lower()]
    snippet = c1_text[:130].replace("\n", " ")
    print(f"  [{iid}]")
    print(f"     C1b: {snippet}")
    print(f"     Gender terms: {hits if hits else '(none)'}")
    print()

# --- True failure rate per category ---
print("=" * 70)
print("D4. TRUE FAILURE RATE PER CATEGORY (corrected for echo + meta)")
print("=" * 70)
c2["true_failure"] = (
    (c2.script_class == "pure_latin")
    | c2.prompt_echo
    | c2.meta_response
)
tab = c2.groupby("category").agg(
    total=("image_id", "count"),
    pure_latin=("script_class", lambda s: (s == "pure_latin").sum()),
    prompt_echo=("prompt_echo", "sum"),
    meta_response=("meta_response", "sum"),
    true_failure=("true_failure", "sum"),
)
tab["true_fail_pct"] = (tab.true_failure / tab.total * 100).round(1)
print(tab.to_string())

# --- Save enriched file ---
c2.to_csv(RES / "C2_enriched.csv", index=False, encoding="utf-8")
print(f"\nWrote results/C2_enriched.csv")