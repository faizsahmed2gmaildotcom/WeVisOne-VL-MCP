# Matting: P3M-500-NP, AM-2K-test, PPM-100

## Download

Please download and extract each dataset into `$RAW` with the following folder names:

| Folder         | Source                                        | Required files                                           |
| -------------- | --------------------------------------------- | -------------------------------------------------------- |
| `P3M-10k/`     | [JizhiziLi/P3M](https://github.com/JizhiziLi/P3M) | `validation/P3M-500-NP/{original_image,mask}`        |
| `AM-2K/`       | [JizhiziLi/GFM](https://github.com/JizhiziLi/GFM) | `validation/{original,mask}` (200 images)            |
| `PPM-100/`     | [ZHKKKe/PPM](https://github.com/ZHKKKe/PPM)       | `{image,matte}`                                      |

## Convert

```bash
python -m evaluation.prepare.matting.prepare_matting --raw_root $RAW --eval_root $EVAL_ROOT
```

The script links each folder into `$EVAL_ROOT`, matches images and alpha mattes
by file name, and writes one row per image:

| Dataset      | Rows | Expression |
| ------------ | ---- | ---------- |
| `P3M-500-NP` | 500  | `person`   |
| `AM-2K-test` | 200  | `animal`   |
| `PPM-100`    | 100  | `person`   |

For these datasets, the model does not predict a box. Instead, the mask decoder
is prompted with the bounding box of the ground-truth alpha (pixels above 127).
