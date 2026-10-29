
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip

from PIL import Image
from tqdm import tqdm

from common import load_annotations, find_images, resolve_image_path
from config import (
    RESULTS_DIR,
    EN_TEXT_COL,
    CATEGORY_COL,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    IMAGE_BATCH_SIZE,
)


def retrieval_metrics(scores, relevant_masks):
    """
    Evaluate retrieval with multiple relevant candidates per query.

    scores: shape (number of queries, number of candidates)
    relevant_masks: boolean array of the same shape.
    """
    order = np.argsort(-scores, axis=1)
    recalls = {k: [] for k in (1, 5, 10)}
    reciprocal_ranks = []

    for i in range(scores.shape[0]):
        ranked_relevance = relevant_masks[i, order[i]]
        relevant_ranks = np.flatnonzero(ranked_relevance)

        if len(relevant_ranks) == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / (relevant_ranks[0] + 1))

        for k in recalls:
            recalls[k].append(
                float(np.any(ranked_relevance[:k]))
            )

    result = {
        f"Recall@{k}": float(np.mean(values))
        for k, values in recalls.items()
    }
    result["MRR"] = float(np.mean(reciprocal_ranks))
    result["Queries"] = int(scores.shape[0])
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and clean the annotations.
    df = load_annotations().copy()

    df[EN_TEXT_COL] = (
        df[EN_TEXT_COL].fillna("").astype(str).str.strip()
    )
    df = df[df[EN_TEXT_COL] != ""].copy()

    image_index = find_images()
    df["resolved_image_path"] = df["image_path"].apply(
        lambda value: resolve_image_path(value, image_index)
    )
    df = df[df["resolved_image_path"].notna()].reset_index(drop=True)

    if len(df) < 2:
        raise RuntimeError("At least two valid image-caption pairs are required.")

    # Report dataset duplication.
    print("=" * 65)
    print("CULTUREALIGN DUPLICATE-AWARE IMAGE-TEXT EVALUATION")
    print("=" * 65)
    print(f"Valid image-caption records: {len(df)}")
    print(f"Unique English captions: {df[EN_TEXT_COL].nunique()}")
    print(
        "Unique Bangla-English pairs: "
        f"{df.drop_duplicates(subset=['alt_text_bn', EN_TEXT_COL]).shape[0]}"
    )

    # Each row is an image-caption record.
    # Identical English captions count as multiple relevant candidates.
    english_values = df[EN_TEXT_COL].to_numpy()
    relevant = (
        english_values[:, None] == english_values[None, :]
    )

    # Load OpenCLIP.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Model: {CLIP_MODEL}")
    print(f"Pretrained weights: {CLIP_PRETRAINED}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()

    # Encode images.
    image_features = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(df), IMAGE_BATCH_SIZE),
            desc="Encoding images",
        ):
            paths = df["resolved_image_path"].iloc[
                start:start + IMAGE_BATCH_SIZE
            ]

            batch_images = []
            for path in paths:
                try:
                    with Image.open(path) as image:
                        batch_images.append(
                            preprocess(image.convert("RGB"))
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"Cannot read image {path}: {error}"
                    ) from error

            batch = torch.stack(batch_images).to(device)
            features = model.encode_image(batch)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            image_features.append(features.cpu())

    image_features = torch.cat(image_features, dim=0)

    # Encode English captions.
    text_features = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(df), 64),
            desc="Encoding English captions",
        ):
            captions = df[EN_TEXT_COL].iloc[start:start + 64].tolist()
            tokens = tokenizer(captions).to(device)

            features = model.encode_text(tokens)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            text_features.append(features.cpu())

    text_features = torch.cat(text_features, dim=0)

    # Similarity matrix: each image compared with every English caption.
    scores = (image_features @ text_features.T).numpy()

    # Exact paired-row relevance for comparison.
    exact_relevance = np.eye(len(df), dtype=bool)

    rows = []
    for direction, matrix in [
        ("Image -> English", scores),
        ("English -> Image", scores.T),
    ]:
        for relevance_name, mask in [
            ("Exact paired row", exact_relevance),
            ("Same English caption", relevant),
        ]:
            metrics = retrieval_metrics(matrix, mask)
            rows.append({
                "Direction": direction,
                "Relevance definition": relevance_name,
                **metrics,
            })

    results = pd.DataFrame(rows)

    print("\n=== RETRIEVAL RESULTS ===")
    print(results.round(4).to_string(index=False))

    output_path = RESULTS_DIR / "image_retrieval_duplicate_aware.csv"
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved results to: {output_path}")
    print(
        "\nInterpretation: 'Same English caption' allows any image-caption "
        "row with an identical English caption to count as relevant. "
        "It does not establish semantic or cultural entity equivalence."
    )


if __name__ == "__main__":
    main()
