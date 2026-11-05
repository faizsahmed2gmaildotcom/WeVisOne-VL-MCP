# WeVisOne-VL MCP

A stdio MCP server that exposes WeVisOne-VL image perception tools to an agent.
The matching skill is at `skill/wevisone-vl/SKILL.md`.

## What it does

Give the agent a local image path and a query. The server runs the appropriate
WeVisOne-VL task, returns the raw model text, and returns rendered images. The
model is loaded for the call and released before the call returns.

## Installation

TODO: installation and client registration instructions.

## Using the skill

Load or mention `wevisone-vl-mcp` when the task involves an image, then ask in
plain language. The agent should pick the tool and shape the query.

Examples:

- "Detect the sofa, lamp and coffee table in `/data/livingroom.jpg`."
- "Where is the boy holding the microphone in `/data/boys.jpg`?"
- "How many white chess pieces are in `/data/chess.jpg`? Use points."
- "Read all the text in `/data/coffee_sign.jpg`."
- "Find the word COFFEE in `/data/coffee_sign.jpg`."
- "Cut out the person in `/data/seabreeze.jpg` with a soft matte."
- "Analyze these images together: detect people in `a.jpg` and matte the dog in `b.jpg`."

## Tools

| Tool | Query | Returns |
|---|---|---|
| `detection` | Newline-separated categories | Boxes + overlay |
| `referring` | One referring expression | Boxes + overlay |
| `point` | Category or phrase | Points + overlay |
| `count_box` | Category or phrase | Boxes, total + overlay |
| `count_point` | Category or phrase | Points, total + overlay |
| `ocr` | None | All words + overlay |
| `ocr_grounding` | Exact text | Matching text + overlay |
| `segmentation` | Category or phrase | Mask + overlay, alpha, cutout |
| `matting` | Category or phrase | Alpha matte + overlay, alpha, cutout |
| `analyze_batch` | List of `{image_path, tool, query}` | Per-image results in request order |

## Query rules in one line each

- `detection`: nouns only, one per line.
- `referring`, `segmentation`, `matting`: one descriptive phrase, not a category list.
- `point`, `count_box`, `count_point`: a category or a phrase.
- `ocr`: no query.
- `ocr_grounding`: the exact text to find.

## Output

Artifacts are written under the agent's current working directory:

```text
wevisone_vl_outputs/<tool>/<timestamp>-<id>/
  response.txt
  result.json
  overlay.png
  alpha.png
  cutout.png
  masks/mask_000.png
```

Object tools return `overlay.png`. Segmentation and matting also return
`alpha.png` and `cutout.png`. The text result always starts with the raw model
response, followed by an `Artifacts:` path. Batch results add an `[i] tool | path`
header per request.

Coordinates in the raw text are integers from 0 to 1000, normalized to the image
size. `None` means the target was rejected as absent. Counting output ends with
`Total number: N`.

## Configuration

Copy `.env.example` to `.env` in the repository root. Process environment
variables override the file.

| Variable | Default | Meaning |
|---|---|---|
| `WEVISONE_MODEL` | `tencent/WeVisOne-VL-2B` | Hugging Face id or local checkpoint |
| `WEVISONE_BACKEND` | `transformers` | `transformers` or `vllm` |
| `WEVISONE_DEVICE` | `auto` | `auto`, `cuda:0`, `npu:0` or `cpu` |
| `WEVISONE_BATCH_LIMIT` | `16` | Maximum images in one `analyze_batch` |
| `WEVISONE_MIN_PIXELS` | `16384` | Minimum vision pixels per image |
| `WEVISONE_MAX_PIXELS` | `655360` | Maximum vision pixels per image |
| `WEVISONE_DECODER_BATCH_SIZE` | `4` | Masks decoded per forward pass |
| `WEVISONE_CNN_SIZE` | `768` | Mask-decoder input resolution |
| `WEVISONE_LOG_LEVEL` | `INFO` | stderr verbosity |

Inference request logs are appended to `skill/logs/requests.jsonl`. Each line
records the timestamp, tool, image path, image size, prompt and batch id.

## Known Issues

### cuDNN aborts when `LD_LIBRARY_PATH` is unset

Symptom: `Invalid handle. Cannot load symbol cudnnGetVersion` followed by a
process abort as soon as CUDA is used.

Cause: cuDNN 9 resolves its split sub-libraries (`libcudnn_ops`, `libcudnn_cnn`,
and so on) by bare name. When the `nvidia/cudnn/lib` directory is not on
`LD_LIBRARY_PATH`, torch cannot resolve them and aborts.

Resolution: the server preloads every `libcudnn*.so*` from
`site-packages/nvidia/cudnn/lib` with `ctypes.CDLL(..., RTLD_GLOBAL)` before
`import torch` (`_preloadCudnn()` in `mcp/wevisone_vl_mcp.py`). No shell setup is
required. If it still fails, launch the server with
`LD_LIBRARY_PATH` pointing at that directory.

### CUDA out of memory on 8 GB GPUs

Symptom: `CUDA out of memory` while generating text or, more often, while
decoding the mask.

Cause: the 2B checkpoint uses about 4.7 GB in bf16, the desktop compositor uses
about 1.3 GB, and the mask decoder's high-resolution refinement needs another
1 GB or more. An 8 GB card runs out of headroom.

Resolution: the defaults are reduced for small GPUs:

- `WEVISONE_MAX_PIXELS=655360` instead of the upstream 2,073,600.
- `WEVISONE_CNN_SIZE=768` instead of 1024.
- `WEVISONE_DECODER_BATCH_SIZE=4`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before torch loads.
- The generation cache is freed (`gc.collect()` and `torch.cuda.empty_cache()`)
  before mask decoding.

Raise `WEVISONE_MAX_PIXELS` to `1280 * 32 * 32` or `2025 * 32 * 32` and
`WEVISONE_CNN_SIZE` to 1024 if the GPU has more free memory; detail improves.
Closing other GPU users (for example Discord) also helps.

### Batch mask decoding is sequential

One `analyze_batch` call shares a single model load and one batched generation
pass, but masks are decoded one image at a time. This avoids the extra peak
memory of decoding every mask in one forward pass. Large batches still take
longer; lower `WEVISONE_BATCH_LIMIT` or split the batch if a call approaches the
client's tool timeout.
