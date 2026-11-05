# Counting: CountQA, PixMo-Count, CLOC, KubriCount, MixCount

## Download

```bash
dl() { huggingface-cli download "$1" --repo-type dataset --local-dir "$RAW/$2" "${@:3}"; }
dl Jayant-Sravan/CountQA    CountQA
dl allenai/pixmo-count      pixmo-count --include "data/validation-*" "data/test-*"
dl liuchang666/KubriCount   KubriCount  --include "testA/*" "testB/*" "metadata/test*" "shards/testA/*" "shards/testB/*"
dl CorentinDumery/MixCount  MixCount    --include "parquet_data/val-*" "parquet_data/test-*"
```

**CLOC** is not available on the Hugging Face Hub. Please prepare its `data/` folder following [count-anything/data/README.md](https://github.com/Mengqi-Lei/count-anything/blob/main/data/README.md), including `images/` and `augmented/` (from `cloc_augmented.zip`), and extract the v1.1 annotations (`cloc_annotations_v1.1.zip`, available from the same links) as follows:

```text
$RAW/CLOC/data/{images,augmented}/...
$RAW/CLOC/annotations/{val,test}_split_expanded_by_class.json   # from cloc_annotations_v1.1.zip
```



## Convert

```bash
python -m evaluation.prepare.counting.prepare_counting --raw_root $RAW --eval_root $EVAL_ROOT
```


| Dataset                                 | Rows        | Notes                                                                                       |
| --------------------------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| `CountQA`                               | 1528        | one row per question; the counting phrase is extracted from the question                    |
| `pixmo-count-val` / `pixmo-count-test`  | 540 / 540   | images are fetched from their URLs (see below)                                              |
| `cloc-val` / `cloc-test`                | 6074 / 9232 | one row per image-class pair; `$EVAL_ROOT/CLOC/{images,augmented}` are symlinks into `$RAW` |
| `KubriCount-testA` / `KubriCount-testB` | 9837 / 9725 | images extracted from the tar shards                                                        |
| `MixCount-val` / `MixCount-test`        | 3309 / 2931 | one row per class in the image                                                              |


Only `gt_count` is used for scoring. `gt` contains the boxes provided by the
release (CLOC, KubriCount exemplars, MixCount) and is empty otherwise.

**PixMo-Count images.** The release only provides image URLs. The script
downloads each image from its URL. For unavailable URLs, it retrieves the
image from the
[UWGZQ/pixmo_images](https://huggingface.co/datasets/UWGZQ/pixmo_images) cache
(`../pixmo_cache.py`), which takes a few minutes per image. The
published `image_sha256` is not checked, since many URLs now return files with
different bytes but identical pixels.