# Referring segmentation: RefCOCO, RefCOCO+, RefCOCOg, ReasonSeg, Ref-ZOM

## Download

```bash
# RefCOCO / RefCOCO+ / RefCOCOg: one JSONL record per expression, masks as COCO RLE
huggingface-cli download PaDT-MLLM/RefCOCO --repo-type dataset --local-dir $RAW/RefCOCO \
  --include "refcoco_val.json" "refcoco_test*.json" "refcoco+_val.json" "refcoco+_test*.json" \
            "refcocog_val.json" "refcocog_test.json"
```

- **ReasonSeg**: download `val.zip` and `test.zip` from
[dvlab-research/LISA](https://github.com/dvlab-research/LISA) and extract them
to `$RAW/ReasonSeg/{val,test}/`.
- **Ref-ZOM**: download `Ref_ZOM.p` and `instances.json` from
[toggle1995/RIS-DMMI](https://github.com/toggle1995/RIS-DMMI) to `$RAW/Ref-ZOM/`.
- **Images**: RefCOCO/+/g use COCO `train2014`, and Ref-ZOM uses COCO `train2014`
and `val2014`. Please place the full [COCO 2014](https://cocodataset.org/#download)
images in `$EVAL_ROOT/coco/train2014/` and `$EVAL_ROOT/coco/val2014/`. Note that
`coco/train2014/` in Rex-Omni-EvalData only contains the images used by RefCOCOg.

## Convert

```bash
python -m evaluation.prepare.refseg.prepare_refseg --raw_root $RAW --eval_root $EVAL_ROOT
```

| Dataset                                    | Rows      | Notes                                                                                         |
| ------------------------------------------ | --------- | --------------------------------------------------------------------------------------------- |
| `RefCOCO_seg` / `RefCOCO+_seg`             | 21586 / 21373 | val + testA + testB                                                                       |
| `RefCOCOg_seg`                             | 14498     | val + test                                                                                    |
| `ReasonSeg_seg_val` / `ReasonSeg_seg_test` | 200 / 779 | first expression only; `$EVAL_ROOT/ReasonSeg` is a symlink into `$RAW`                        |
| `Ref-ZOM_seg_test`                         | 21770     | one row per sentence; all referred instances are merged into one mask (empty for `zero` rows) |

**ReasonSeg masks** are generated in the same way as the LISA evaluator:
polygons are drawn in descending order of area, shapes labeled `flag` are
skipped, and shapes whose label contains `ignore` are stored in `ignore_rle` and
excluded from IoU. Short phrases (`is_sentence: false`) are converted to
`What is {phrase} in this image?`.
