---
name: wevisone
description: Use WeVisOne-VL image perception tools via MCP.
version: 0.1.0
author: faizsahmed, DeepSeek V4.1 Flash + OpenCode + JetBrains Air
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MCP, Vision, Detection, Segmentation, Matting, OCR, Counting]
    related_skills: []
---

# WeVisOne-VL MCP Skill

Use the WeVisOne-VL MCP server to analyze a local image with detection, referring
localization, pointing, counting, OCR, segmentation or alpha matting. The server
returns the model's raw text plus rendered result images.

## When to Use

- The user supplies or points to an image and asks what or where something is.
- The user asks to count, locate, box, point at, segment or cut out an object.
- The user asks to read text from an image or find a specific word.
- The user asks for a mask, alpha matte, transparent cutout or background removal.
- Several images need different analyses and should share one model load.

Don't use for:

- Generating or editing an image.
- Reading an image the user has not made available as a local file.
- General visual question answering that does not fit a tool below.

## Prerequisites

- The `wevisone-vl` MCP server is registered in the client.
- The image exists at a local path. Relative paths resolve against the agent cwd.
- Tool names may appear prefixed with the server name in the client; use the names
  below as the suffix.

## Tool Selection

| User intent | Tool | `query` |
|---|---|---|
| Find all instances of one or more categories | `detection` | Newline-separated nouns |
| Find one object described by a phrase | `referring` | One referring expression |
| Mark object centers | `point` | Category or phrase |
| Count with boxes | `count_box` | Category or phrase |
| Count with points | `count_point` | Category or phrase |
| Read all text | `ocr` | None |
| Find a known text string | `ocr_grounding` | Exact text |
| Get a binary mask | `segmentation` | Category or phrase |
| Get a soft alpha matte or cutout | `matting` | Category or phrase |
| Mix tools across several images | `analyze_batch` | One request per image |

Selection rules:

- A plain category ("sofa", "person") can use `detection`, `point`, `count_box`,
  `count_point`, `segmentation` or `matting`.
- A relational or attribute phrase ("the boy holding the microphone") must use
  `referring`, `segmentation` or `matting`, never `detection`.
- Use `point` when the user wants locations but not boxes.
- Use `segmentation` for a hard mask and `matting` for hair, fur or transparent
  edges. Both return overlay, alpha and cutout images.
- Use `ocr_grounding` only when the text is already known. Use `ocr` to discover it.

## Prompt Construction

### detection

Pass categories only, one per line. Do not write sentences or include "all".

```text
sofa
pillow
coffee table
lamp
```

### referring, segmentation, matting

Pass one descriptive phrase. Add the attribute or relation that distinguishes the
target from other objects of the same class.

```text
the person in the red jacket on the left
```

### point, count_box, count_point

Pass a category or a referring phrase. Prefer plural categories for counting.

```text
white chess pieces
```

### ocr

Pass no query. The server uses the fixed all-words prompt.

### ocr_grounding

Pass the exact text to locate, preserving spelling and punctuation.

```text
COFFEE
```

### analyze_batch

Pass one object per image, each with its own `image_path`, `tool` and `query`.
Use it when the tools differ or when several images should share one model load.
The whole call fails if any path is missing or any image is corrupted.

```json
{
  "requests": [
    {"image_path": "/data/room.jpg", "tool": "detection", "query": "sofa\nlamp"},
    {"image_path": "/data/person.jpg", "tool": "matting", "query": "person"}
  ]
}
```

## Procedure

1. Confirm the image is a local file and note its absolute path.
2. Choose the tool from the table above.
3. Build the query with the rules for that tool.
4. Call the tool or `analyze_batch`.
5. Read the raw text first, then inspect the returned overlay or alpha image.
6. Report object counts and locations to the user in plain language.

## Reading the Raw Output

- Boxes: `<ref>label</ref><box><x1><y1><x2><y2></box>`.
- Points: `<ref>label</ref><point><x><y></point>`.
- Coordinates are integers from 0 to 1000, normalized to image width and height.
  Multiply x by width / 1000 and y by height / 1000 to get pixels.
- Multiple objects are comma-separated inside one shape tag.
- `None` means the model rejected the query because the target is absent.
  Refine the phrase once; do not loop on the same query.
- Counting tools end with `Total number: N`.
- Segmentation and matting add `<OBJ>` after each box and produce alpha masks.

## Output and Artifacts

- Every call returns the untouched raw model text and an `Artifacts:` path.
- Object tools return `overlay.png`.
- Segmentation and matting return `overlay.png`, `alpha.png` and `cutout.png`.
- Files live under `<agent cwd>/wevisone_vl_outputs/<tool>/<timestamp>-<id>/`.
- `result.json` contains the prompt, response, latency and parsed predictions.
- The model is unloaded after every call. Use `analyze_batch` to amortize that cost.

## Pitfalls

- Do not pass a JSON array as a query. Detection and pointing take newline-separated
  text.
- Do not use `detection` for a relational phrase; the model expects categories there.
- Do not invent categories that the user did not ask for; extra categories add noise.
- A batch over `WEVISONE_BATCH_LIMIT` is rejected before inference.
- Coordinates are not pixels. Convert them before reporting pixel positions.
- A returned overlay can look unchanged when the target is absent; check the raw
  text for `None` before claiming success.

## Verification

- The raw text is non-empty.
- The expected image block is present (overlay for object tools; overlay, alpha and
  cutout for pixel tools).
- The artifact path exists and contains `result.json`.
- For counting, the reported count matches `Total number` and the visible boxes or
  points.
