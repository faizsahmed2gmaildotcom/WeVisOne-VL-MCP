# Pointing: PointBench, PixMo-Points-Eval

## Download

Both downloads are pinned to the revisions used in the paper. The script
checks the archive and parquet SHA-256 digests.

```bash
huggingface-cli download PointArena/pointarena-data --repo-type dataset \
  --revision 77ec5dca697b25025e655e2ec34fd2207856924c --local-dir $RAW/PointBench \
  --include data.json pixmo_metadata.csv selected_images.zip selected_masks.zip
huggingface-cli download allenai/pixmo-points-eval --repo-type dataset \
  --revision e237f0491ca49bf8dec04e06fc0c3807a37ef157 --local-dir $RAW/PixMo-Points-Eval
```

## Convert

```bash
python -m evaluation.prepare.pointing.prepare_pointing --raw_root $RAW --eval_root $EVAL_ROOT
```

| Dataset             | Rows | Notes                                                                                  |
| ------------------- | ---- | -------------------------------------------------------------------------------------- |
| `PointBench`        | 979  | 982 rows in `data.json`; rows 56, 454 and 789 (0-based) are dropped because their mask size differs from the image |
| `PixMo-Points-Eval` | 436  | question `Point to {label}.`; `benchmark_slice` is set by the number of targets        |

- **PointBench.** The archives are extracted to `$EVAL_ROOT/PointBench/files/`.
Mask pixels with any channel above 127 are treated as foreground. For steerable
questions, the reference point from `pixmo_metadata.csv` is appended to the
question. Counting questions use the `point_counting_union_region` protocol and
are not scored.
- **PixMo-Points-Eval.** The parquet provides image URLs and SHA-256 digests.
  Images are saved as `$EVAL_ROOT/PixMo-Points-Eval/images-official/<sha256>.img`
and must match the digest. Each image is downloaded from its URL, or retrieved
  from the [UWGZQ/pixmo_images](https://huggingface.co/datasets/UWGZQ/pixmo_images)
cache (`../pixmo_cache.py`) if the URL is unavailable, which
  takes a few minutes per image. All-zero placeholder masks are removed, so
  queries for absent objects have no targets.
