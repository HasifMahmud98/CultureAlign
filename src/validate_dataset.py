
from common import load_annotations, find_images, resolve_image_path


def main():
    df = load_annotations()
    image_index = find_images()

    resolved_paths = [
        resolve_image_path(value, image_index)
        for value in df["image_path"]
    ]

    matched = sum(path is not None for path in resolved_paths)
    missing_images = len(resolved_paths) - matched

    print("=" * 50)
    print("SHILPOBANGLA DATASET VALIDATION")
    print("=" * 50)
    print(f"Annotation rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Images discovered: {len(image_index)}")
    print(f"Annotations matched to images: {matched}")
    print(f"Missing image matches: {missing_images}")
    print(f"Missing Bangla text: {df['alt_text_bn'].isna().sum()}")
    print(f"Missing English text: {df['eng_text'].isna().sum()}")
    print(f"Missing category: {df['class_name'].isna().sum()}")
    print("\nCategory counts:")
    print(df["class_name"].value_counts(dropna=False).to_string())

    if missing_images:
        print("\nExamples of unmatched image paths:")
        for value, path in zip(df["image_path"], resolved_paths):
            if path is None:
                print(f"  {value}")
                missing_images -= 1
                if missing_images <= 4:
                    break

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
