# Referring detection: Ref-Adv-s, PR-Bench, DVGBench, Ref-L4, RefCOCOg-rej

## Download

```bash
dl() { huggingface-cli download "$1" --repo-type dataset --local-dir "$RAW/$2" "${@:3}"; }
dl dddraxxx/ref-adv-s     Ref-Adv-s    --revision e7a53e352b5885b8228fc6afa8645ab78e76d5f1
dl thisis1go/PR-Bench     PR-Bench     --revision d6fefc0cd779afddc5c83c8603cb34ae4f6c7e79
dl erenzhou/DVGBench      DVGBench     --revision 9b708ce1ee7df0aa6bddbde7c59aaeaa895cdbd0
dl JierunChen/Ref-L4      Ref-L4       --include "ref-l4-*.parquet" "images.tar.gz"
dl lirang04/RefCOCOg_rej  RefCOCOg_rej --include "data/validation-*.parquet"
```

The script reads:

```text
$RAW/Ref-Adv-s/data/*.parquet                      # images embedded
$RAW/PR-Bench/{annotation.jsonl,images.zip}
$RAW/DVGBench/{dvg_test.jsonl,images.zip}
$RAW/Ref-L4/{ref-l4-val.parquet,ref-l4-test.parquet,images.tar.gz}
$RAW/RefCOCOg_rej/data/validation-*.parquet        # images embedded
```

## Convert

```bash
python -m evaluation.prepare.referring.prepare_referring --raw_root $RAW --eval_root $EVAL_ROOT
```

| Dataset            | Rows  | Notes                                                                |
| ------------------ | ----- | -------------------------------------------------------------------- |
| `Ref-Adv-s`        | 1142  | the expression is the `normal_caption` field                         |
| `PR-Bench`         | 6000  | `task_type == "reject"` rows have no target box                      |
| `DVGBench`         | 873   | the expression is the `question` field                               |
| `Ref-L4-all`       | 45341 | val + test, sorted by `id`; boxes converted from xywh                |
| `RefCOCOg-rej-val` | 11490 | the expression is the `rejected_exp` field, which never has a target, so `gt` is empty |

All rows use `task_name: referring_object_detection` and the expression as the
single category.
