# WeVisOne-VL Demo & Cookbooks

This directory contains the interactive Gradio web interface (`app.py`) and three end-to-end workflow notebooks for **WeVisOne-VL**.

## 1. Interactive Gradio Web Demo

Launch from the repository root:

```bash
# Default Hugging Face checkpoint
CUDA_VISIBLE_DEVICES=0 python demo/app.py --model tencent/WeVisOne-VL-4B

# High-throughput vLLM backend on custom port
CUDA_VISIBLE_DEVICES=0 python demo/app.py \
  --model tencent/WeVisOne-VL-4B \
  --backend vllm \
  --server-name 0.0.0.0 \
  --server-port 8080
```

### Available CLI Options

- `--model`: Hugging Face model ID or local checkpoint directory (default: `tencent/WeVisOne-VL-4B` or `$WEVISONE_MODEL`).
- `--backend`: Inference backend (`transformers` or `vllm`, default: `transformers`).
- `--device`: Runtime device (`auto`, `cuda:0`, `npu:0`, or `cpu`).
- `--server-name`: Host address to bind (default: `0.0.0.0`).
- `--server-port`: Port number (default: `8080`).
- `--share`: Enable a public Gradio share link.
- `--mock`: Skip model loading and return a fake matte (UI preview only).

## 2. Cookbooks

Interactive Jupyter notebook tutorials are located in `../tutorials`:

- [Object-level tutorial](../tutorials/01_object_level_pipeline.ipynb): Grounding, Dense Detection, Box Counting, Point Counting, Pointing.
- [OCR tutorial](../tutorials/02_ocr_pipeline.ipynb): End-to-end OCR, Text Grounding.
- [Pixel-level tutorial](../tutorials/03_segmentation_matting_pipeline.ipynb): Segmentation, Matting.
