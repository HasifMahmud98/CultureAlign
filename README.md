# CultureAlign — Local ShilpoBangla Pilot

Local Windows project for exploratory Bangla-English text retrieval and image-English caption retrieval using ShilpoBangla.

## 1. Put the data in place

Extract the dataset into `data/ShilpoBangla/`, keeping the original annotation CSV and image folders.

## 2. Create the Conda environment

Open Anaconda Prompt in the project folder:

```bash
conda env create -f environment.yml
conda activate culturealign
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
```

## 3. Open in VS Code

Open this folder in VS Code. Install Microsoft's Python extension if it is not already installed. Use `Ctrl+Shift+P` → `Python: Select Interpreter` → select the `culturealign` Conda environment.

In the VS Code terminal, activate the environment if needed:

```bash
conda activate culturealign
```

## 4. Inspect the dataset

```bash
python src/inspect_dataset.py
```

If there is more than one CSV, edit `src/config.py` and set `CSV_PATH` to the annotation CSV. For example:

```python
CSV_PATH = DATA_DIR / "annotations.csv"
```

If the column names differ, update `IMAGE_COL`, `BN_TEXT_COL`, `EN_TEXT_COL`, and `CATEGORY_COL` in `src/config.py`.

## 5. Validate data

```bash
python src/validate_dataset.py
```

Check the printed image-found count and the output `results/dataset_validation.csv`.

## 6. Run the text baseline

```bash
python src/run_text_baseline.py
```

Outputs:
- `results/culturealign_bilingual_retrieval.csv`
- `results/text_top5_predictions.csv`

## 7. Run the image baseline

```bash
python src/run_image_baseline.py
```

Outputs:
- `results/culturealign_image_retrieval.csv`
- `results/image_top5_predictions.csv`

The first model run downloads model weights and requires internet. Later runs can reuse the local cache.

## Important scientific limitation

These are paired-caption/image retrieval pilot experiments. They are **not** an entity-alignment benchmark and do not prove CultureAlign improves entity alignment. The diagonal-positive setup assumes row i's caption/image is the unique correct match for row i. Duplicate captions or multiple valid matches can bias metrics. A proper entity-alignment study needs two independently sourced entity collections, verified gold links, and a held-out test split.

## Troubleshooting

- `No CSV found`: confirm the dataset is extracted under `data/ShilpoBangla/`.
- `Multiple CSV files found`: set `CSV_PATH` in `src/config.py`.
- `Missing expected columns`: run `inspect_dataset.py` and update column names in `src/config.py`.
- Images missing: inspect the CSV image paths and folder layout. The loader tries the dataset root and then searches by filename.
- Out of memory: lower `IMAGE_BATCH_SIZE` in `src/config.py` (try 8) and rerun.
- Model download fails: check internet access, then rerun. Models download on first use.
