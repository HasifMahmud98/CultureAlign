# src/generate_C1b_en_glm.py
import os, base64, time
from pathlib import Path
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "results" / "sample_manifest.csv"
OUT      = PROJECT_ROOT / "results" / "generated_C1b_en.csv"

client = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4"  # OpenAI-compatible endpoint
)

MODEL = "glm-4.6v-flash"

PROMPT_EN = (
    "Write a short, factual description of this cultural object for an alt-text. "
    "State what it is, its material or technique if identifiable, its likely region or "
    "community of origin, and how it is used. Do not add introductory phrases. "
    "If you are uncertain about a detail, omit it rather than guess."
)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

manifest = pd.read_csv(MANIFEST)
rows, errors = [], 0

for _, r in tqdm(manifest.iterrows(), total=len(manifest)):
    try:
        b64 = encode_image(r["image_path"])
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_EN},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            temperature=0.2,
            max_tokens=120
        )
        gen = resp.choices[0].message.content.strip()
    except Exception as e:
        gen = f"<ERROR: {type(e).__name__}: {e}>"
        errors += 1
    rows.append({
        "image_id": r["image_id"],
        "category": r["category"],
        "condition": "C1b_en_glm",
        "language": "en",
        "generated_text": gen,
    })
    time.sleep(0.3)  # be polite to the endpoint

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {len(rows)} rows to {OUT}   Errors: {errors}")