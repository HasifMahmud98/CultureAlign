
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
    IMAGE_COL,
    EN_TEXT_COL,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    IMAGE_BATCH_SIZE,
)


def retrieval_metrics(scores, relevance):
    """Calculate retrieval metrics when each query can have multiple matches."""
    ranks = []

    for i in range(scores.shape[0]):
        relevant_indices = np.flatnonzero(relevance[i])

        if len(relevant_indices) == 0:
            ranks.append(float("inf"))
            continue

        order = np.argsort(-scores[i])
        relevant_ranks = np.flatnonzero(
            np.isin(order, relevant_indices)
        )

        ranks.append(int(relevant_ranks[0]) + 1)

    ranks = np.asarray(ranks, dtype=float)

    result = {
        f"Recall@{k}": float(np.mean(ranks <= k))
        for k in (1, 5, 10)
    }
    result["MRR"] = float(
        np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))
    )
    result["Queries"] = len(ranks)
    result["QueriesWithoutMatch"] = int(np.isinf(ranks).sum())

    return result


def main():
    print("=" * 68)
    print("CULTUREALIGN UNIQUE-ITEM RETRIEVAL EVALUATION")
    print("=" * 68)

    # 1. Load annotations and resolve paths
    df = load_annotations().copy()
    df = df.dropna(subset=[EN_TEXT_COL, IMAGE_COL]).copy()

    df[EN_TEXT_COL] = df[EN_TEXT_COL].astype(str).str.strip()
    df = df[df[EN_TEXT_COL] != ""].reset_index(drop=True)

    print("\nIndexing images...")
    image_index = find_images()

    df["resolved_image_path"] = df[IMAGE_COL].apply(
        lambda value: resolve_image_path(value, image_index)
    )

    df = df[
        df["resolved_image_path"].apply(
            lambda value: value is not None and Path(value).is_file()
        )
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid image-caption records were found.")

    # Each row represents an observed image-caption association.
    pairs = df[
        ["resolved_image_path", EN_TEXT_COL]
    ].drop_duplicates().reset_index(drop=True)

    unique_images = sorted(pairs["resolved_image_path"].unique().tolist())
    unique_captions = sorted(pairs[EN_TEXT_COL].unique().tolist())

    image_to_index = {value: i for i, value in enumerate(unique_images)}
    caption_to_index = {value: i for i, value in enumerate(unique_captions)}

    # Relevance matrix: true when the dataset contains this image-caption pair.
    relevance = np.zeros(
        (len(unique_images), len(unique_captions)),
        dtype=bool,
    )

    for row in pairs.itertuples(index=False, name=None):
        image_path, caption = row
        relevance[
            image_to_index[image_path],
            caption_to_index[caption],
        ] = True

    print(f"Valid annotation records: {len(df)}")
    print(f"Unique images: {len(unique_images)}")
    print(f"Unique English captions: {len(unique_captions)}")
    print(f"Unique image-caption associations: {len(pairs)}")
    print(f"Image-caption relevance links: {int(relevance.sum())}")

    # 2. Load the same model as the original baseline
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"Model: {CLIP_MODEL}")
    print(f"Pretrained weights: {CLIP_PRETRAINED}")
    print("\nLoading OpenCLIP...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()

    # 3. Encode each unique image once
    image_batches = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(unique_images), IMAGE_BATCH_SIZE),
            desc="Encoding unique images",
        ):
            paths = unique_images[start:start + IMAGE_BATCH_SIZE]
            batch_images = []

            for path in paths:
                with Image.open(path) as image:
                    batch_images.append(
                        preprocess(image.convert("RGB"))
                    )

            tensor = torch.stack(batch_images).to(device)
            features = model.encode_image(tensor)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            image_batches.append(features.cpu())

    image_features = torch.cat(image_batches, dim=0)

    # 4. Encode each unique English caption once
    text_batches = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(unique_captions), 64),
            desc="Encoding unique captions",
        ):
            captions = unique_captions[start:start + 64]
            tokens = tokenizer(captions).to(device)

            features = model.encode_text(tokens)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            text_batches.append(features.cpu())

    text_features = torch.cat(text_batches, dim=0)

    # 5. Evaluate both directions using the same relevance links
    print("\nCalculating similarity scores...")
    scores = (image_features @ text_features.T).numpy()

    image_to_text = retrieval_metrics(scores, relevance)
    text_to_image = retrieval_metrics(scores.T, relevance.T)

    rows = [
        {
            "Direction": "Image -> English",
            "Gallery": "Unique English captions",
            **image_to_text,
        },
        {
            "Direction": "English -> Image",
            "Gallery": "Unique images",
            **text_to_image,
        },
    ]

    results = pd.DataFrame(rows)

    print("\n=== UNIQUE-ITEM RETRIEVAL RESULTS ===")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "image_retrieval_unique_items.csv"
    results.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")
    print("\nEvaluation completed.")
    print(
        "Note: relevance is based on observed image-caption links, "
        "not verified cultural entity equivalence."
    )


if __name__ == "__main__":
    main()
