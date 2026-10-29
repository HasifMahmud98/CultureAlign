# src/generate_C1b_C2_qwen.py
import torch
import pandas as pd
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "results" / "sample_manifest.csv"
OUT_EN   = PROJECT_ROOT / "results" / "generated_C1b_en.csv"
OUT_BN   = PROJECT_ROOT / "results" / "generated_C2_bn.csv"

# --- Device selection ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
print(f"Device: {DEVICE}, dtype: {DTYPE}")

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
print(f"Loading model: {MODEL_NAME}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, low_cpu_mem_usage=True,
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")

PROMPT_EN = (
    "Write a short, factual description of this cultural object for an alt-text. "
    "State what it is, its material or technique if identifiable, its likely region or "
    "community of origin, and how it is used. Do not add introductory phrases. "
    "If you are uncertain about a detail, omit it rather than guess."
)

PROMPT_BN = (
    "এই সাংস্কৃতিক বস্তুটির জন্য একটি সংক্ষিপ্ত, তথ্যমূলক বর্ণনা বাংলায় লেখো। "
    "এটি কী, সম্ভব হলে উপাদান বা কৌশল, কোন অঞ্চল বা সম্প্রদায়ের, এবং কীভাবে ব্যবহৃত হয় — "
    "যতটা নিশ্চিত ততটাই লেখো। ভূমিকা যোগ করো না। অনিশ্চিত হলে সেই বিবরণ বাদ দাও।"
)

@torch.no_grad()
def generate(image_path, prompt, max_new_tokens=100):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
            {"type": "text",  "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    imgs, vids = process_vision_info(messages)
    inputs = processor(
        text=[text], images=imgs, videos=vids, return_tensors="pt"
    ).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

manifest = pd.read_csv(MANIFEST)
print(f"Loaded {len(manifest)} sample rows.")

CONDITIONS = [
    ("C1b_en_qwen2vl", "en", PROMPT_EN, OUT_EN),
    ("C2_bn_qwen2vl",  "bn", PROMPT_BN, OUT_BN),
]

for cond_name, lang, prompt, out_path in CONDITIONS:
    print(f"\n=== Running condition: {cond_name} ===")
    rows, errors = [], 0
    t0 = time.time()
    for _, r in tqdm(manifest.iterrows(), total=len(manifest)):
        try:
            gen = generate(r["image_path"], prompt)
            if not gen:
                gen = "<EMPTY>"
        except Exception as e:
            gen = f"<ERROR: {type(e).__name__}: {e}>"
            errors += 1
        rows.append({
            "image_id":       r["image_id"],
            "category":       r["category"],
            "condition":      cond_name,
            "language":       lang,
            "generated_text": gen,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(f"Wrote {out_path}   Errors: {errors}   Elapsed: {elapsed:.1f}s")

print("\nDONE.")