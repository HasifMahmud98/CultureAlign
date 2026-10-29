
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common import load_annotations

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def recall_at_k(similarity, k):
    k = min(k, similarity.shape[1])
    top_k = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    targets = np.arange(similarity.shape[0])
    return np.mean(np.any(top_k == targets[:, None], axis=1))


def main():
    df = load_annotations().copy()

    for col in ["alt_text_bn", "eng_text"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[
        (df["alt_text_bn"] != "") &
        (df["eng_text"] != "")
    ].copy()

    # Keep one record for each exact bilingual caption pair.
    unique_df = df.drop_duplicates(
        subset=["alt_text_bn", "eng_text"]
    ).reset_index(drop=True)

    print(f"Original bilingual records: {len(df)}")
    print(f"Unique bilingual caption pairs: {len(unique_df)}")

    model = SentenceTransformer(MODEL_NAME)

    bn_embeddings = model.encode(
        unique_df["alt_text_bn"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    en_embeddings = model.encode(
        unique_df["eng_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    # Each Bangla caption retrieves its paired English caption.
    bn_to_en = cosine_similarity(bn_embeddings, en_embeddings)
    en_to_bn = cosine_similarity(en_embeddings, bn_embeddings)

    print("\n=== UNIQUE-PAIR RETRIEVAL ===")
    for direction, similarity in [
        ("Bangla -> English", bn_to_en),
        ("English -> Bangla", en_to_bn),
    ]:
        print(f"\n{direction}")
        for k in (1, 5, 10):
            print(f"Recall@{k}: {recall_at_k(similarity, k):.4f}")

    print(
        "\nNote: This is a diagnostic evaluation on unique caption pairs, "
        "not a verified cultural entity-alignment benchmark."
    )


if __name__ == "__main__":
    main()
