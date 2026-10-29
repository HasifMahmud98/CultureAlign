
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip

from PIL import Image
from tqdm import tqdm

from common import load_annotations, resolve_image_path, find_images


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# Use project config values when available; otherwise defaults.
try:
    from config import CLIP_MODEL, CLIP_PRETRAINED, IMAGE_BATCH_SIZE
except ImportError:
    CLIP_MODEL = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"
    IMAGE_BATCH_SIZE = 32

EN_TEXT_COL = "eng_text"


# --------------------------------------------------
# Retrieval evaluation
# --------------------------------------------------

def retrieval_metrics(scores):
    """
    Evaluate paired image-text retrieval.
    Each row's correct paired item has the same index.
    """
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(scores.shape[0], dtype=int)

    for i in range(scores.shape[0]):
        matches = np.flatnonzero(order[i] == i)

        if len(matches) == 0:
            raise RuntimeError(f"Could not find target for query {i}")

        ranks[i] = matches[0] + 1

    return {
        "Recall@1": float(np.mean(ranks <= 1)),
        "Recall@5": float(np.mean(ranks <= 5)),
        "Recall@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
        "N": int(len(ranks)),
    }


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset annotations.
    df = load_annotations().copy()

    df[EN_TEXT_COL] = (
        df[EN_TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Find image files and resolve annotation paths.
    image_index = find_images()

    df["resolved_image_path"] = df["image_path"].apply(
        lambda value: resolve_image_path(value, image_index)
    )

    # Remove rows without an image or English caption.
    df = df[
        df["resolved_image_path"].notna()
        & (df[EN_TEXT_COL] != "")
    ].reset_index(drop=True)

    print("=" * 60)
    print("CULTUREALIGN IMAGE-TEXT RETRIEVAL BASELINE")
    print("=" * 60)
    print(f"Valid image-caption pairs: {len(df)}")
    print(f"CLIP model: {CLIP_MODEL}")
    print(f"Pretrained weights: {CLIP_PRETRAINED}")

    if len(df) < 2:
        raise RuntimeError(
            "At least two valid image-caption pairs are required."
        )

    # Select computation device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load OpenCLIP.
    print("\nLoading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
    )

    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()

    # --------------------------------------------------
    # Encode images
    # --------------------------------------------------

    image_batches = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(df), IMAGE_BATCH_SIZE),
            desc="Encoding images",
        ):
            paths = df["resolved_image_path"].iloc[
                start:start + IMAGE_BATCH_SIZE
            ]

            images = []

            for path in paths:
                try:
                    with Image.open(path) as image:
                        images.append(
                            preprocess(image.convert("RGB"))
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to read image '{path}': {error}"
                    ) from error

            batch = torch.stack(images).to(device)

            features = model.encode_image(batch)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            image_batches.append(features.cpu())

    image_features = torch.cat(image_batches, dim=0)

    # --------------------------------------------------
    # Encode English captions
    # --------------------------------------------------

    text_batches = []
    text_batch_size = 64

    with torch.no_grad():
        for start in tqdm(
            range(0, len(df), text_batch_size),
            desc="Encoding English captions",
        ):
            texts = df[EN_TEXT_COL].iloc[
                start:start + text_batch_size
            ].tolist()

            tokens = tokenizer(texts).to(device)

            features = model.encode_text(tokens)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            text_batches.append(features.cpu())

    text_features = torch.cat(text_batches, dim=0)

    # --------------------------------------------------
    # Calculate similarity and retrieval metrics
    # --------------------------------------------------

    scores = (image_features @ text_features.T).numpy()

    results = pd.DataFrame([
        {
            "Direction": "Image -> English",
            **retrieval_metrics(scores),
        },
        {
            "Direction": "English -> Image",
            **retrieval_metrics(scores.T),
        },
    ])

    print("\n" + "=" * 60)
    print("EXPLORATORY IMAGE-TEXT RETRIEVAL RESULTS")
    print("=" * 60)
    print(results.round(4).to_string(index=False))

    # Save summary metrics.
    metrics_path = RESULTS_DIR / "culturealign_image_retrieval.csv"
    results.to_csv(metrics_path, index=False)

    # --------------------------------------------------
    # Save top-5 image-to-English retrieval examples
    # --------------------------------------------------

    top_k = min(5, len(df))
    prediction_rows = []

    for i in range(len(df)):
        candidate_indices = np.argsort(-scores[i])[:top_k]

        for rank, j in enumerate(candidate_indices, start=1):
            prediction_rows.append({
                "query_row": i,
                "query_category": df.iloc[i]["class_name"],
                "query_image": str(
                    df.iloc[i]["resolved_image_path"]
                ),
                "rank": rank,
                "candidate_row": int(j),
                "candidate_category": df.iloc[j]["class_name"],
                "candidate_english_caption": df.iloc[j][EN_TEXT_COL],
                "is_paired_row": bool(i == j),
                "cosine_similarity": float(scores[i, j]),
            })

    predictions_path = RESULTS_DIR / "image_top5_predictions.csv"

    pd.DataFrame(prediction_rows).to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved files:")
    print(f"Metrics: {metrics_path}")
    print(f"Top-5 predictions: {predictions_path}")

    print(
        "\nIMPORTANT: These results measure paired image-caption "
        "retrieval, not verified cultural entity alignment."
    )


if __name__ == "__main__":
    main()
