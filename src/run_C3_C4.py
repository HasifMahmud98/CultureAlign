# src/run_C3_C4.py
import sys, torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MODEL = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE).eval()

def translate_batch(texts, src, tgt, batch_size=16):
    out_all = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"{src}->{tgt}"):
        batch = texts[i:i+batch_size]
        tok.src_lang = src
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        bos = tok.convert_tokens_to_ids(tgt)
        with torch.no_grad():
            out = mdl.generate(**enc, forced_bos_token_id=bos, max_length=128)
        out_all.extend(tok.batch_decode(out, skip_special_tokens=True))
    return out_all

c1 = pd.read_csv(RES / "generated_C1b_en.csv", encoding="utf-8")
c2 = pd.read_csv(RES / "generated_C2_bn.csv", encoding="utf-8")

# C3: English (C1b) -> Bangla
print("\n=== C3: English -> Bangla ===")
c3_texts = translate_batch(c1.generated_text.fillna("").tolist(), "eng_Latn", "ben_Beng")
c3 = pd.DataFrame({"image_id": c1.image_id, "category": c1.category,
                   "condition": "C3_en2bn", "language": "bn",
                   "source_en": c1.generated_text,
                   "generated_text": c3_texts})
c3.to_csv(RES / "generated_C3_en2bn.csv", index=False, encoding="utf-8")

# C4: Bangla (C2) -> English
print("\n=== C4: Bangla -> English ===")
c4_texts = translate_batch(c2.generated_text.fillna("").tolist(), "ben_Beng", "eng_Latn")
c4 = pd.DataFrame({"image_id": c2.image_id, "category": c2.category,
                   "condition": "C4_bn2en", "language": "en",
                   "source_bn": c2.generated_text,
                   "generated_text": c4_texts})
c4.to_csv(RES / "generated_C4_bn2en.csv", index=False, encoding="utf-8")

print("\n" + "="*70)
print("C3 SAMPLE (5 rows)")
print("="*70)
for _, r in c3.head(5).iterrows():
    print(f"\n[{r.image_id}] ({r.category})")
    print(f"  EN source: {r.source_en[:200]}")
    print(f"  BN trans : {r.generated_text[:200]}")

print("\n" + "="*70)
print("C4 SAMPLE (10 rows — the critical check)")
print("="*70)
for _, r in c4.head(10).iterrows():
    print(f"\n[{r.image_id}] ({r.category})")
    print(f"  BN source: {r.source_bn[:200]}")
    print(f"  EN trans : {r.generated_text[:200]}")

print("\nWrote generated_C3_en2bn.csv and generated_C4_bn2en.csv")