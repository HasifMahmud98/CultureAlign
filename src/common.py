
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ShilpoBangla"
RESULTS_DIR = PROJECT_DIR / "results"

ANNOTATION_FILE = DATA_DIR / "Image_alt_text.xlsx"


def load_annotations():
    """Load ShilpoBangla annotations from Excel or CSV."""
    if ANNOTATION_FILE.exists():
        df = pd.read_excel(ANNOTATION_FILE)
    else:
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No annotation file found in {DATA_DIR}. "
                "Expected Image_alt_text.xlsx or a CSV file."
            )
        df = pd.read_csv(csv_files[0])

    df.columns = df.columns.astype(str).str.strip()

    required = {"class_name", "image_path", "alt_text_bn", "eng_text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def find_images():
    """Index JPG, JPEG, and PNG images, including nested category folders."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_index = {}

    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_index[path.name.lower()] = path

    return image_index


def resolve_image_path(image_value, image_index=None):
    """Resolve an annotation's image filename or relative path."""
    if pd.isna(image_value):
        return None

    image_value = str(image_value).strip().replace("\\", "/")
    candidate = Path(image_value)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    relative_candidate = DATA_DIR / image_value
    if relative_candidate.exists():
        return relative_candidate

    if image_index is None:
        image_index = find_images()

    return image_index.get(candidate.name.lower())
