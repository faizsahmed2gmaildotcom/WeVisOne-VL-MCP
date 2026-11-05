"""Stdio MCP server exposing WeVisOne-VL image perception tools."""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import sys
import sysconfig
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from mcp.server.mcpserver import Image as McpImage
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent
from PIL import Image as PILImage
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "skill"
LOG_DIR = SKILL_ROOT / "logs"
LOG_PATH = LOG_DIR / "requests.jsonl"
OUTPUT_ROOT_NAME = "wevisone_vl_outputs"
DEFAULT_MODEL_ID = "tencent/WeVisOne-VL-2B"
PIXEL_TOOLS = ("segmentation", "matting")
SUPPORTED_TOOLS = (
    "detection",
    "referring",
    "point",
    "count_box",
    "count_point",
    "ocr",
    "ocr_grounding",
    "segmentation",
    "matting",
)
ARTIFACT_KEYS_BY_TOOL = {
    tool: ("overlay", "alpha", "cutout") if tool in PIXEL_TOOLS else ("overlay",)
    for tool in SUPPORTED_TOOLS
}

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _preloadCudnn() -> None:
    """Load cuDNN 9 sub-libraries before torch needs them by bare name."""
    # shortcut: cuDNN 9 resolves its sub-libraries without a path, so torch aborts
    # when LD_LIBRARY_PATH is unset; preload them here. Revisit on torch/cuDNN upgrades.
    try:
        lib_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cudnn" / "lib"
        for path in sorted(lib_dir.glob("libcudnn*.so*")):
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
    except (KeyError, OSError):
        return


_preloadCudnn()

import torch  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from wevisone_vl import WeVisOneVL, build_prompt  # noqa: E402
from wevisone_vl.engine import SEG_RE, InferenceResult  # noqa: E402
from wevisone_vl.visualization import save_result  # noqa: E402


def _positiveIntEnv(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back on bad values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.getLogger("wevisone_vl_mcp").warning(
            "Ignoring non-integer %s=%r; using %d", name, raw, default
        )
        return default
    if value < 1:
        logging.getLogger("wevisone_vl_mcp").warning(
            "Ignoring non-positive %s=%r; using %d", name, raw, default
        )
        return default
    return value


def _resolveLogLevel(raw: str) -> int:
    """Map a level name to a logging level, defaulting to INFO."""
    level = getattr(logging, raw.upper(), None)
    return level if isinstance(level, int) else logging.INFO


MODEL_ID = os.environ.get("WEVISONE_MODEL", DEFAULT_MODEL_ID)
BACKEND = os.environ.get("WEVISONE_BACKEND", "transformers")
DEVICE = os.environ.get("WEVISONE_DEVICE", "auto")
BATCH_LIMIT = _positiveIntEnv("WEVISONE_BATCH_LIMIT", 16)
MIN_PIXELS = _positiveIntEnv("WEVISONE_MIN_PIXELS", 16 * 32 * 32)
MAX_PIXELS = _positiveIntEnv("WEVISONE_MAX_PIXELS", 640 * 32 * 32)
DECODER_BATCH_SIZE = _positiveIntEnv("WEVISONE_DECODER_BATCH_SIZE", 4)
CNN_SIZE = _positiveIntEnv("WEVISONE_CNN_SIZE", 768)
LOG_LEVEL = _resolveLogLevel(os.environ.get("WEVISONE_LOG_LEVEL", "INFO"))

ToolName = Literal[
    "detection",
    "referring",
    "point",
    "count_box",
    "count_point",
    "ocr",
    "ocr_grounding",
    "segmentation",
    "matting",
]

logger = logging.getLogger("wevisone_vl_mcp")
_CALL_LOCK = threading.Lock()


def _setupLogger() -> None:
    """Send diagnostics to stderr so stdout stays a clean MCP protocol stream."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False


class BatchRequest(BaseModel):
    """One image and the operation to run on it."""

    image_path: str = Field(description="Absolute path to a local image file.")
    tool: ToolName = Field(description="WeVisOne-VL operation to run on this image.")
    query: str = Field(
        default="",
        description=(
            "Tool-specific query. Newline-separated categories for detection, point, "
            "count_box and count_point; one expression for referring, segmentation and "
            "matting; the exact text for ocr_grounding; empty for ocr."
        ),
    )


@dataclass
class ResolvedRequest:
    """A batch request whose image path has been resolved and verified."""

    request: BatchRequest
    image_path: Path


def _resolveRequests(requests: list[BatchRequest]) -> list[ResolvedRequest]:
    """Resolve paths and reject missing or corrupted images before model load."""
    resolved: list[ResolvedRequest] = []
    for request in requests:
        path = Path(request.image_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise ToolError(f"Image not found: {path}")
        try:
            with PILImage.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ToolError(f"Corrupted or unreadable image: {path}") from exc
        resolved.append(ResolvedRequest(request=request, image_path=path))
    return resolved


def _loadImages(resolved: list[ResolvedRequest]) -> list[PILImage.Image]:
    """Fully decode every image to RGB before any inference starts."""
    images: list[PILImage.Image] = []
    for item in resolved:
        try:
            with PILImage.open(item.image_path) as source:
                images.append(source.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise ToolError(f"Corrupted or unreadable image: {item.image_path}") from exc
    return images


def _buildPrompts(resolved: list[ResolvedRequest], images: list[PILImage.Image]) -> list[str]:
    """Build the exact prompt each image will be sent with."""
    return [
        build_prompt(item.request.tool, item.request.query, image_size=image.size)
        for item, image in zip(resolved, images)
    ]


def _writeRequestLog(
    resolved: list[ResolvedRequest],
    images: list[PILImage.Image],
    prompts: list[str],
    batch_id: str,
) -> None:
    """Append one JSONL record per image and mirror it to stderr."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        for item, image, prompt in zip(resolved, images, prompts):
            record = {
                "timestamp": timestamp,
                "batch_id": batch_id,
                "tool": item.request.tool,
                "image_path": str(item.image_path),
                "image_size": list(image.size),
                "prompt": prompt,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                "request batch=%s tool=%s image=%s prompt=%s",
                batch_id,
                item.request.tool,
                item.image_path,
                prompt,
            )


def _getEngine() -> WeVisOneVL:
    """Load & cache the shared engine on first use."""
    if not getattr(_getEngine, "engine"):
        logger.info("Loading %s (backend=%s, device=%s)", MODEL_ID, BACKEND, DEVICE)
        _getEngine.engine = WeVisOneVL(
            MODEL_ID,
            backend=BACKEND,
            device=DEVICE,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            decoder_batch_size=DECODER_BATCH_SIZE,
        )
        _getEngine.engine.cnn_size = CNN_SIZE
    return _getEngine.engine


def _releaseEngine() -> None:
    """Drop the engine and return its VRAM to the GPU."""
    if not hasattr(_getEngine, "engine"):
        return
    _getEngine.engine = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Released WeVisOne-VL model")


def _predict(images: list[PILImage.Image], prompts: list[str]):
    """Generate text, free the generation cache, then decode masks one image at a time."""
    engine = _getEngine()
    if len(images) == 1:
        responses = [engine.generate_text(images[0], prompts[0])]
    else:
        responses = engine.generate_text_batch(images, prompts)
    gc.collect()
    torch.cuda.empty_cache()
    results = []
    for image, prompt, response in zip(images, prompts, responses):
        boxes: list[list[float]] = []
        masks: list = []
        if SEG_RE.search(response):
            gc.collect()
            torch.cuda.empty_cache()
            boxes, masks = engine.decode_masks(image, prompt, response)
        results.append(
            InferenceResult(response=response, prompt=prompt, boxes=boxes, masks=masks)
        )
    return results


def _createOutputDir(tool: str) -> Path:
    """Create a unique artifact directory under the agent's cwd."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path.cwd() / OUTPUT_ROOT_NAME / tool / f"{stamp}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _saveResults(
    resolved: list[ResolvedRequest],
    images: list[PILImage.Image],
    results: list,
) -> list[dict[str, Path]]:
    """Write every raw response, overlay and mask artifact for the call."""
    return [
        save_result(_createOutputDir(item.request.tool), image, result)
        for item, image, result in zip(resolved, images, results)
    ]


def _buildContent(
    resolved: list[ResolvedRequest],
    results: list,
    saved: list[dict[str, Path]],
) -> list[TextContent | McpImage]:
    """Render raw text and images in request order."""
    multiple = len(resolved) > 1
    blocks: list[TextContent | McpImage] = []
    for index, (item, result, paths) in enumerate(zip(resolved, results, saved)):
        if multiple:
            blocks.append(
                TextContent(
                    type="text",
                    text=f"[{index + 1}] {item.request.tool} | {item.image_path}",
                )
            )
        blocks.append(TextContent(type="text", text=result.response))
        blocks.append(
            TextContent(type="text", text=f"Artifacts: {paths['metadata'].parent}")
        )
        for key in ARTIFACT_KEYS_BY_TOOL[item.request.tool]:
            blocks.append(McpImage(path=paths[key]))
    return blocks


def _analyzeRequests(requests: list[BatchRequest]) -> list[TextContent | McpImage]:
    """Validate, infer, save and unload for one tool call."""
    if not requests:
        raise ToolError("At least one image request is required.")
    if len(requests) > BATCH_LIMIT:
        raise ToolError(
            f"Batch contains {len(requests)} images; the limit is {BATCH_LIMIT}. "
            "Raise WEVISONE_BATCH_LIMIT or split the batch."
        )
    with _CALL_LOCK:
        resolved = _resolveRequests(requests)
        images = _loadImages(resolved)
        prompts = _buildPrompts(resolved, images)
        _writeRequestLog(resolved, images, prompts, uuid.uuid4().hex[:12])
        try:
            results = _predict(images, prompts)
            saved = _saveResults(resolved, images, results)
            return _buildContent(resolved, results, saved)
        except ToolError:
            raise
        except Exception as exc:
            logger.exception("Inference failed")
            raise ToolError(f"WeVisOne-VL inference failed: {exc}") from exc
        finally:
            _releaseEngine()


mcpServer = MCPServer(
    name="wevisone-vl",
    instructions=(
        "Analyze a local image with WeVisOne-VL. Pick detection for a list of categories, "
        "referring for one referring expression, point for object points, count_box or "
        "count_point for counts, ocr for all text, ocr_grounding for a known text string, "
        "segmentation for binary masks and matting for alpha mattes. Use analyze_batch to "
        "mix operations over several images in one call."
    ),
)

_setupLogger()


@mcpServer.tool(structured_output=False)
def detection(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Locate every instance of one or more categories.

    Pass categories as newline-separated text, for example "sofa\\npillow\\nlamp".
    Returns the raw model text plus an overlay image.
    """
    return _analyzeRequests(
        [BatchRequest(image_path=image_path, tool="detection", query=query)]
    )


@mcpServer.tool(structured_output=False)
def referring(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Locate one object described by a referring expression.

    Use a descriptive phrase such as "the boy holding the microphone".
    Returns the raw model text plus an overlay image.
    """
    return _analyzeRequests(
        [BatchRequest(image_path=image_path, tool="referring", query=query)]
    )


@mcpServer.tool(structured_output=False)
def point(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Mark the center of each matching object with a point.

    Pass categories as newline-separated text. Returns the raw model text plus
    an overlay image.
    """
    return _analyzeRequests([BatchRequest(image_path=image_path, tool="point", query=query)])


@mcpServer.tool(structured_output=False)
def count_box(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Count matching objects and report them as boxes.

    The raw text ends with a "Total number" line. Returns the raw model text
    plus an overlay image.
    """
    return _analyzeRequests(
        [BatchRequest(image_path=image_path, tool="count_box", query=query)]
    )


@mcpServer.tool(structured_output=False)
def count_point(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Count matching objects and report them as points.

    The raw text ends with a "Total number" line. Returns the raw model text
    plus an overlay image.
    """
    return _analyzeRequests(
        [BatchRequest(image_path=image_path, tool="count_point", query=query)]
    )


@mcpServer.tool(structured_output=False)
def ocr(image_path: str) -> list[TextContent | McpImage]:
    """Detect all words in the image as boxes.

    Takes no query. Returns the raw model text plus an overlay image.
    """
    return _analyzeRequests([BatchRequest(image_path=image_path, tool="ocr")])


@mcpServer.tool(structured_output=False)
def ocr_grounding(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Locate one known text string in the image.

    Pass the exact text to find. Returns the raw model text plus an overlay image.
    """
    return _analyzeRequests(
        [BatchRequest(image_path=image_path, tool="ocr_grounding", query=query)]
    )


@mcpServer.tool(structured_output=False)
def segmentation(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Segment one object described by a category or referring expression.

    Returns the raw model text plus overlay, alpha and cutout images.
    """
    return _analyzeRequests(
        [BatchRequest(image_path=image_path, tool="segmentation", query=query)]
    )


@mcpServer.tool(structured_output=False)
def matting(image_path: str, query: str) -> list[TextContent | McpImage]:
    """Produce a soft alpha matte for one object.

    Returns the raw model text plus overlay, alpha and cutout images.
    """
    return _analyzeRequests(
        [BatchRequest(image_path=image_path, tool="matting", query=query)]
    )


@mcpServer.tool(structured_output=False)
def analyze_batch(requests: list[BatchRequest]) -> list[TextContent | McpImage]:
    """Run different WeVisOne-VL operations over several images in one model load.

    Each item specifies image_path, tool and query. Items are returned in request
    order, grouped as header, raw text, artifact path and images. The whole call
    fails if any image is missing or corrupted.
    """
    return _analyzeRequests(requests)


def main() -> None:
    """Run the WeVisOne-VL MCP server over stdio."""
    mcpServer.run(transport="stdio")


if __name__ == "__main__":
    main()
