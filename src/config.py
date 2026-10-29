from pathlib import Path

# Project root is the parent of src/
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "ShilpoBangla"
RESULTS_DIR = PROJECT_DIR / "results"

# After running inspect_dataset.py, paste the actual CSV path here if needed.
# Example: DATA_DIR / "annotations.csv"
CSV_PATH = None

# These are the expected published column names. Update them if your CSV differs.
IMAGE_COL = "image_path"
BN_TEXT_COL = "alt_text_bn"
EN_TEXT_COL = "eng_text"
CATEGORY_COL = "class_name"

# OpenCLIP settings
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Batch sizes: lower them if you run out of RAM/VRAM.
TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 32
