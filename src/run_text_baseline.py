
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common import load_annotations

SEED = 42
TEST_RATIO = 0.2
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def recall_at_k(similarity, targets, k):
    """Recall@k with target indices in the full retrieval gallery."""
    k = min(k, similarity.shape[1])
    top_k = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    return np.mean(np.any(top_k == targets[:, None], axis=1))


def main():
    df = load_annotations().copy()

    df["alt_text_bn"] = df["alt_text_bn"].fillna("").astype(str).str.strip()
    df["eng_text"] = df["eng_text"].fillna("").astype(str).str.strip()
    df = df[
        (df["alt_text_bn"] != "") &
        (df["eng_text"] != "")
    ].reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Not enough bilingual text pairs to evaluate.")

    print(f"Usable bilingual pairs: {len(df)}")
    print(f"Loading model: {MODEL_NAME}")

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

   
    # Fixed query set; search the full 1,200-caption gallery.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(df))
    split = max(1, int(len(indices) * (1 - TEST_RATIO)))
    test_indices = indices[split:]

    bn_test = bn_embeddings[test_indices]
    en_test = en_embeddings[test_indices]

    # Each query's correct partner is at its original dataset index.
    bn_to_en = cosine_similarity(bn_test, en_embeddings)
    en_to_bn = cosine_similarity(en_test, bn_embeddings)

    print("\n=== Bangla -> English Text Retrieval ===")
    for k in (1, 5, 10):
        score = recall_at_k(bn_to_en, test_indices, k)
        print(f"Recall@{k}: {score:.4f}")

    print("\n=== English -> Bangla Text Retrieval ===")
    for k in (1, 5, 10):
        score = recall_at_k(en_to_bn, test_indices, k)
        print(f"Recall@{k}: {score:.4f}")

    print(f"\nTest queries: {len(test_indices)}")
    print(f"Candidate gallery: {len(df)}")
    print("Evaluation: paired-caption retrieval, full-gallery search")



if __name__ == "__main__":
    main()
