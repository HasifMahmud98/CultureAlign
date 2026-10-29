
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from common import load_annotations

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    df = load_annotations().copy()
    df["alt_text_bn"] = df["alt_text_bn"].fillna("").astype(str).str.strip()
    df["eng_text"] = df["eng_text"].fillna("").astype(str).str.strip()
    df = df[
        (df["alt_text_bn"] != "") &
        (df["eng_text"] != "")
    ].reset_index(drop=True)

    model = SentenceTransformer(MODEL_NAME)

    bn_embeddings = model.encode(
        df["alt_text_bn"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    en_embeddings = model.encode(
        df["eng_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    # Inspect a reproducible sample of 20 pairs.
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(df), size=min(20, len(df)), replace=False
    )

    # Correct paired-caption cosine similarity.
    paired_similarities = np.sum(
        bn_embeddings * en_embeddings, axis=1
    )

    print("\n=== RANDOM CAPTION-PAIR INSPECTION ===")
    for idx in sample_indices:
        row = df.iloc[idx]
        print(f"\nIndex: {idx} | Category: {row['class_name']}")
        print(f"Bangla: {row['alt_text_bn']}")
        print(f"English: {row['eng_text']}")
        print(
            "Paired cosine similarity: "
            f"{paired_similarities[idx]:.4f}"
        )

    print("\n=== PAIRED SIMILARITY SUMMARY ===")
    print(f"Mean:   {paired_similarities.mean():.4f}")
    print(f"Median: {np.median(paired_similarities):.4f}")
    print(f"Min:    {paired_similarities.min():.4f}")
    print(f"Max:    {paired_similarities.max():.4f}")

    output_path = "../results/caption_pair_inspection.csv"
    from pathlib import Path
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result_df = df.copy()
    result_df["paired_cosine_similarity"] = paired_similarities
    result_df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nSaved inspection results to: {output.resolve()}")


if __name__ == "__main__":
    main()
