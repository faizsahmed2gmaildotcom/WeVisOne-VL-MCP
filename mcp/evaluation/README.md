# Evaluation

This directory provides the evalution code for WeVisOne-VL.

```text
evaluation/
├── inference_text_prompt.py   # vLLM: detection, referring, OCR, counting, pointing
├── inference_refseg.py        # Transformers + mask decoder: referring segmentation
├── inference_matting.py       # Transformers + mask decoder: image matting
├── fastevaluate/              # numpy port of the FastEval box matcher (COCO / LVIS)
├── metrics/
│   ├── coco_lvis_metric.py    # COCO, LVIS
│   ├── other_metric.py        # dense / referring detection, OCR, pointing
│   ├── counting_metric.py     # CountQA, PixMo-Count, CLOC, KubriCount, MixCount
│   ├── pointing_bench_metric.py  # PointBench, PixMo-Points-Eval
│   ├── refseg_metric.py       # RefCOCO/+/g, ReasonSeg, Ref-ZOM
│   └── matting_metric.py      # P3M-500-NP, AM-2K-test, PPM-100, WeRefMat
├── scripts/
│   ├── eval_coco.sh  eval_lvis.sh  eval_others.sh
│   └── eval_refseg.sh  eval_matting.sh
├── prepare/                   # download instructions and conversion scripts for other benchmarks
│   └── referring/  counting/  pointing/  refseg/  matting/
└── utils/
    ├── parser.py  prompt_builder.py
    └── convert_coco_lvis_to_standard_format.py
```



## Setup

```bash
pip install -e ".[vllm]"          # vllm==0.23.0, transformers==5.12.1, qwen-vl-utils==0.0.14
pip install pycocotools opencv-python shapely
```

`inference_refseg.py` and `inference_matting.py` default to
`--attn_implementation flash_attention_2`; pass `sdpa` if flash-attn is not installed.

## Data

Due to copyright restrictions, we cannot provide the evaluation data directly.
Please download it from the sources below and organize it as described in
[Data layout](#data-layout).

### Download



#### Rex-Omni-EvalData

The following benchmarks are provided in [Mountchicken/Rex-Omni-EvalData](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData) and can be used directly:

- Box evaluation: COCO, LVIS, Dense200, VisDrone, RefCOCOg val/test, HumanRef,
HierText, IC15, TotalText and SROIE.
- Point evaluation: COCO, LVIS, Dense200, VisDrone, HumanRef and RefCOCOg val/test.
- The LVIS ground truth `missing_annotaitons/lvis_v1_val_with_filename2.json`
(the folder name follows the original release).

```bash
huggingface-cli download Mountchicken/Rex-Omni-EvalData --repo-type dataset --local-dir $EVAL_ROOT
cd $EVAL_ROOT && for f in *.tar.gz; do tar -xzf "$f"; done
```



#### Other benchmarks

Please download them from their official sources and convert them with the scripts in [prepare/](prepare/). 


| Folder                                  | Benchmarks                                          |
| --------------------------------------- | --------------------------------------------------- |
| [prepare/referring](prepare/referring/) | Ref-Adv-s, PR-Bench, DVGBench, Ref-L4, RefCOCOg-rej |
| [prepare/counting](prepare/counting/)   | CountQA, PixMo-Count, CLOC, KubriCount, MixCount    |
| [prepare/pointing](prepare/pointing/)   | PointBench, PixMo-Points-Eval                       |
| [prepare/refseg](prepare/refseg/)       | RefCOCO, RefCOCO+, RefCOCOg, ReasonSeg, Ref-ZOM     |
| [prepare/matting](prepare/matting/)     | P3M-500-NP, AM-2K-test, PPM-100                     |


```bash
python -m evaluation.prepare.referring.prepare_referring --raw_root $RAW --eval_root $EVAL_ROOT
```

Note that `coco/train2014/` in Rex-Omni-EvalData only contains the images used
by RefCOCOg. RefCOCO, RefCOCO+ and Ref-ZOM require the full
[COCO 2014](https://cocodataset.org/#download) `train2014/` and `val2014/`
folders under `$EVAL_ROOT/coco/`.

### Data layout

Please organize the data as follows. The number after each annotation file is its expected number of rows. The evaluation scripts read `$EVAL_ROOT/_annotations/<type>/<DATASET>.jsonl`, and all image paths in the annotation files are relative to `$EVAL_ROOT`.

```text
$EVAL_ROOT/
├── _annotations/
│   ├── box_eval/
│   │   ├── COCO.jsonl (4952)  LVIS.jsonl (19626)  Dense200.jsonl (200)  VisDrone.jsonl (1610)
│   │   ├── RefCOCOg_val.jsonl (4889)  RefCOCOg_test.jsonl (9577)  HumanRef.jsonl (5000)
│   │   ├── HierText.jsonl (1723)  IC15.jsonl (496)  TotalText.jsonl (300)  SROIE.jsonl (360)
│   │   ├── Ref-Adv-s.jsonl (1142)  PR-Bench.jsonl (6000)  DVGBench.jsonl (873)
│   │   ├── Ref-L4-all.jsonl (45341)  RefCOCOg-rej-val.jsonl (11490)
│   │   ├── CountQA.jsonl (1528)  pixmo-count-val.jsonl (540)  pixmo-count-test.jsonl (540)
│   │   ├── cloc-val.jsonl (6074)  cloc-test.jsonl (9232)
│   │   ├── KubriCount-testA.jsonl (9837)  KubriCount-testB.jsonl (9725)
│   │   └── MixCount-val.jsonl (3309)  MixCount-test.jsonl (2931)
│   ├── point_eval/
│   │   ├── COCO.jsonl (4950)  LVIS.jsonl (19615)  Dense200.jsonl (200)  VisDrone.jsonl (1610)
│   │   ├── HumanRef.jsonl (4997)  RefCOCOg_val.jsonl (4887)  RefCOCOg_test.jsonl (9572)
│   │   └── PointBench.jsonl (979)  PixMo-Points-Eval.jsonl (436)
│   ├── refseg/
│   │   ├── RefCOCO_seg.jsonl (21586)  RefCOCO+_seg.jsonl (21373)  RefCOCOg_seg.jsonl (14498)
│   │   └── ReasonSeg_seg_val.jsonl (200)  ReasonSeg_seg_test.jsonl (779)  Ref-ZOM_seg_test.jsonl (21770)
│   └── matting/
│       └── P3M-500-NP.jsonl (500)  AM-2K-test.jsonl (200)  PPM-100.jsonl (100)  WeRefMat.jsonl (not yet released)
├── coco/                       # instances_val2017.json, val2017/, train2017/, train2014/ + val2014/ (full)
├── missing_annotaitons/        # lvis_v1_val_with_filename2.json
├── dense200/  visdrone/  humanref/  hiertext/  ic15/  totaltext/  sroie/
├── Ref-Adv-s/  PR-Bench/  DVGBench/  Ref-L4/  RefCOCOg-rej/
├── CountQA/  pixmo-count-val/  pixmo-count-test/  CLOC/
├── KubriCount-testA/  KubriCount-testB/  MixCount-val/  MixCount-test/
├── PointBench/  PixMo-Points-Eval/  ReasonSeg/
└── P3M-10k/  AM-2K/  PPM-100/
```

Image folders from Rex-Omni-EvalData keep their original names (e.g. `coco/val2017/000000000139.jpg`). The remaining folders are created by the `prepare` scripts (e.g. `Ref-Adv-s/images/0000.jpg`).

### Annotation formats

The `prepare` scripts already produce the formats below. This section is a reference for adding new benchmarks.

**Box annotations** (`box_eval/`). One line per image. Boxes are `[x0, y0, x1, y1]` in absolute pixels.

```json
{"image_path": "coco/val2017/000000000139.jpg", "dataset_name": "COCO",
 "task_name": "common_object_detection", "categories": ["tv", "chair"],
 "gt": {"tv": [[7.0, 167.8, 156.4, 262.6]], "chair": [[359.0, 218.1, 414.5, 316.9]]}}
```

- `dataset_name`: the benchmark name. It determines the prompt (see
`utils/prompt_builder.py`).
- `categories`: the queries sent to the model. Detection queries each category,
OCR uses `"text line"`, and referring and counting use the expression.
- `task_name`: determines how predicted category names are matched to the ground
truth. For `common_object_detection`, `long_tailed_object_detection` and
`dense_object_detection`, only letters are compared, case-insensitively (e.g.
`traffic light` matches `traffic_light`). For other tasks, case, spaces and
underscores are ignored. Use `referring_object_detection` for referring and
`counting` for counting.
- For referring expressions without a target (RefCOCOg-rej and PR-Bench
rejection queries), the box list is empty, e.g.
`"gt": {"A yellow and red fire hydrant.": []}`.
- Counting annotations additionally contain `gt_count`, which is used for
scoring. Their boxes may be empty (e.g. CountQA).

**Point annotations** (`point_eval/`). Same as box annotations, with an
additional `gt_mask` field containing one COCO RLE mask per instance. A
predicted point is correct if it falls inside an instance mask. PointBench and
PixMo-Points-Eval additionally contain:

- `question`: the original question of the benchmark, used to build the prompt.
- `sample_id`, `image_width`, `image_height`, and `gt_count` (the number of masks).
- `evaluation_protocol`: only rows with `"point_mask_f1"` are scored.
- `benchmark_slice`: the subset used for per-slice results.

**Referring segmentation annotations** (`refseg/`). One line per expression.

```json
{"id": 376848, "image_path": "coco/train2014/COCO_train2014_000000376848.jpg",
 "expression": "person furthest to the right", "gt_rle": {"size": [491, 640], "counts": "..."},
 "split": "val", "dataset_name": "RefCOCO_seg"}
```

- `split`: results are reported per split (RefCOCO val/testA/testB, ReasonSeg val/test).
- ReasonSeg additionally contains `ignore_rle` (pixels excluded from IoU) and `is_sentence`.
- Ref-ZOM additionally contains `source_type` (`zero`, `one` or `many`). For
`zero`, `gt_rle` is an empty mask.

**Matting annotations** (`matting/`). One line per image.

```json
{"sample_id": "p_00355abf", "dataset": "P3M-500-NP",
 "image_path": "P3M-10k/validation/P3M-500-NP/original_image/p_00355abf.jpg",
 "alpha_path": "P3M-10k/validation/P3M-500-NP/mask/p_00355abf.png", "expression": "person"}
```

For images with multiple alpha mattes, use `mattes` (a list of paths, merged by
pixel-wise max) instead of `alpha_path`. If a matte covers only part of the
image, its position is given by `alpha_box_xywh`.

## Run

The scripts run one model replica on each visible GPU (`NUM_GPUS` limits the
number of GPUs) and save `answer.jsonl` and `eval_results.json` to
`$OUTPUT_ROOT/<DATASET>/`.

```bash
export MODEL_PATH=checkpoints/WeVisOne-VL-4B EVAL_ROOT=/path/to/eval_data

# detection
bash evaluation/scripts/eval_coco.sh
bash evaluation/scripts/eval_lvis.sh
bash evaluation/scripts/eval_others.sh Dense200 VisDrone
# referring
bash evaluation/scripts/eval_others.sh RefCOCOg_val RefCOCOg_test HumanRef \
  Ref-Adv-s PR-Bench DVGBench Ref-L4-all RefCOCOg-rej-val
# counting
bash evaluation/scripts/eval_others.sh CountQA pixmo-count-val pixmo-count-test cloc-val cloc-test \
  KubriCount-testA KubriCount-testB MixCount-val MixCount-test
# pointing
bash evaluation/scripts/eval_others.sh PointBench PixMo-Points-Eval \
  COCO_point LVIS_point Dense200_point VisDrone_point HumanRef_point RefCOCOg_val_point RefCOCOg_test_point
# ocr
bash evaluation/scripts/eval_others.sh HierText IC15 TotalText SROIE
# seg
bash evaluation/scripts/eval_refseg.sh RefCOCO_seg RefCOCO+_seg RefCOCOg_seg \
  ReasonSeg_seg_val ReasonSeg_seg_test Ref-ZOM_seg_test
# box-prompted matting
bash evaluation/scripts/eval_matting.sh P3M-500-NP AM-2K-test PPM-100
# text-prompted matting (not released now)
bash evaluation/scripts/eval_matting.sh WeRefMat
```

