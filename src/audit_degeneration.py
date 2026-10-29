# src/audit_degeneration.py
import sys, re
import pandas as pd
from pathlib import Path
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"
c2 = pd.read_csv(RES / "C2_enriched.csv", encoding="utf-8")

# Bengali + Latin token splitter (word-level, includes punctuation as separators)
TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z]+")

def max_word_repetition(text, n=4, min_len=3):
    """Return the highest count of any contiguous n-gram token repeated anywhere."""
    if not isinstance(text, str):
        return 0, ""
    toks = TOKEN_RE.findall(text.lower())
    if len(toks) < n:
        return 0, ""
    # count contiguous n-grams globally
    grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0, ""
    counts = Counter(grams)
    top, cnt = counts.most_common(1)[0]
    phrase = " ".join(top)
    return cnt, phrase

def is_degenerate(text):
    if not isinstance(text, str) or len(text) < 40:
        return False, ""
    cnt, phrase = max_word_repetition(text, n=3)
    if cnt >= 3:  # any trigram appearing 3+ times
        return True, phrase
    # single-word repetition check
    words = TOKEN_RE.findall(text.lower())
    if len(words) >= 8:
        wc = Counter(words)
        top_word, top_cnt = wc.most_common(1)[0]
        # if a single word >25% of all words and appears 4+ times
        if top_cnt >= 4 and top_cnt / len(words) > 0.25:
            return True, f"[word] {top_word} (x{top_cnt})"
    return False, ""

c2[["degenerate", "repeated_phrase"]] = c2["generated_text"].apply(
    lambda t: pd.Series(is_degenerate(t))
)

print("=" * 70)
print("R1. DEGENERATION DETECTION (any phrase repeated 3+ times)")
print("=" * 70)
print(f"Total rows:            {len(c2)}")
print(f"Pure Latin (collapse): {(c2.script_class=='pure_latin').sum()}")
print(f"Prompt echo:           {c2.prompt_echo.sum()}")
print(f"Tautology (সাং. বস্তু): {c2.generated_text.str.count('সাংস্কৃতিক বস্তু').ge(2).sum()}")
print(f"Degenerate repetition: {c2.degenerate.sum()}")
print(f"Meta response:         {c2.meta_response.sum()}")

# Rows that are pure_bangla AND non-degenerate AND non-echo AND non-tautology
clean_bn = c2[
    (c2.script_class == "pure_bangla")
    & (~c2.prompt_echo)
    & (~c2.degenerate)
    & (c2.generated_text.str.count("সাংস্কৃতিক বস্তু") < 2)
]
print(f"\n>>> CLEAN BANGLA DESCRIPTIONS: {len(clean_bn)} / {len(c2)} "
      f"({len(clean_bn)/len(c2)*100:.1f}%)")

print("\nBy category:")
print(clean_bn.groupby("category").size().to_string())

# Show a few clean ones so we can verify they're genuinely good
print("\n--- 8 clean Bangla descriptions to inspect ---")
for _, r in clean_bn.head(8).iterrows():
    print(f"\n[{r.image_id}] ({r.category})")
    print(f"  {r.generated_text[:220]}")

# Combined "usable output" rate
total_usable = len(clean_bn)
print("\n" + "=" * 70)
print("R2. HEADLINE NUMBERS FOR THE PAPER")
print("=" * 70)
print(f"Total sample: {len(c2)}")
print(f"Usable Bangla: {total_usable}  ({total_usable/len(c2)*100:.1f}%)")
print(f"Failure:       {len(c2)-total_usable}  ({(len(c2)-total_usable)/len(c2)*100:.1f}%)")

# Save the clean subset
clean_bn.to_csv(RES / "C2_usable_bangla.csv", index=False, encoding="utf-8")
print(f"\nWrote results/C2_usable_bangla.csv")