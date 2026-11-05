# Preparing the benchmarks not in Rex-Omni-EvalData

Each folder covers one benchmark family. Its README says where to download the
releases, and its script converts them into the annotation format described in
[../README.md](../README.md#annotation-formats).

| Folder                     | Benchmarks                                                    | Output                        |
| -------------------------- | ------------------------------------------------------------- | ----------------------------- |
| [referring](referring/)    | Ref-Adv-s, PR-Bench, DVGBench, Ref-L4, RefCOCOg-rej           | `_annotations/box_eval/`      |
| [counting](counting/)      | CountQA, PixMo-Count, CLOC, KubriCount, MixCount              | `_annotations/box_eval/`      |
| [pointing](pointing/)      | PointBench, PixMo-Points-Eval                                 | `_annotations/point_eval/`    |
| [refseg](refseg/)          | RefCOCO, RefCOCO+, RefCOCOg, ReasonSeg, Ref-ZOM               | `_annotations/refseg/`        |
| [matting](matting/)        | P3M-500-NP, AM-2K-test, PPM-100                               | `_annotations/matting/`       |

Every script is run from the repository root and takes the same arguments:

```bash
python -m evaluation.prepare.<folder>.prepare_<folder> --raw_root $RAW --eval_root $EVAL_ROOT
```

- `--raw_root`: the directory containing the downloaded releases, one sub-folder per benchmark.
- `--eval_root`: the evaluation data root used by `..`.
- `--datasets NAME ...` (optional): convert only these datasets; by default, all of the folder.

- Images are saved to `$EVAL_ROOT/<benchmark>/`; large image folders are
symlinked instead. All `image_path` values are relative to `$EVAL_ROOT`.
- When a script is run again, existing images are skipped and only the annotation files are rewritten.
- Each README lists the expected number of rows for verification.
