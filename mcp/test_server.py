"""Real end-to-end test for the WeVisOne-VL MCP server.

Run with the project conda environment:
    python mcp/test_server.py

The test boots the real stdio server, calls the real 2B model on demo images,
checks returned text and images, and verifies VRAM is released between calls.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = Path(__file__).with_name("wevisone_vl_mcp.py")
IMAGE_DIR = PROJECT_ROOT / "demo" / "examples" / "images"
LOG_PATH = PROJECT_ROOT / "skill" / "logs" / "requests.jsonl"
EXPECTED_TOOLS = {
    "detection",
    "referring",
    "point",
    "count_box",
    "count_point",
    "ocr",
    "ocr_grounding",
    "segmentation",
    "matting",
    "analyze_batch",
}
MODEL_VRAM_LIMIT_MIB = 2500


def _gpuUsedMiB() -> int | None:
    """Return used GPU memory, or None when nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def _texts(result) -> list[str]:
    """Return all text blocks of a tool result."""
    return [block.text for block in result.content if block.type == "text"]


def _images(result) -> list:
    """Return all image blocks of a tool result."""
    return [block for block in result.content if block.type == "image"]


async def _checkTools(session: ClientSession) -> None:
    """Verify the advertised tool surface."""
    result = await session.list_tools()
    names = {tool.name for tool in result.tools}
    assert names == EXPECTED_TOOLS, f"unexpected tools: {names}"
    print(f"tools ok: {len(names)}")


async def _checkSingle(session: ClientSession) -> None:
    """Run matting on the demo image and verify text, images and artifacts."""
    result = await session.call_tool(
        "matting",
        {"image_path": str(IMAGE_DIR / "seabreeze.jpg"), "query": "person"},
        read_timeout_seconds=600,
    )
    assert not result.is_error, _texts(result)
    texts = _texts(result)
    assert texts and texts[0].strip(), "empty raw response"
    assert any(text.startswith("Artifacts:") for text in texts), texts
    assert len(_images(result)) == 3, "matting must return overlay, alpha and cutout"
    print("matting ok")


async def _checkBatch(session: ClientSession) -> None:
    """Run mixed operations over two images in one call."""
    result = await session.call_tool(
        "analyze_batch",
        {
            "requests": [
                {
                    "image_path": str(IMAGE_DIR / "livingroom.jpg"),
                    "tool": "detection",
                    "query": "sofa\nlamp",
                },
                {
                    "image_path": str(IMAGE_DIR / "seabreeze.jpg"),
                    "tool": "segmentation",
                    "query": "person",
                },
            ]
        },
        read_timeout_seconds=600,
    )
    assert not result.is_error, _texts(result)
    texts = _texts(result)
    assert any(text.startswith("[1] detection |") for text in texts), texts
    assert any(text.startswith("[2] segmentation |") for text in texts), texts
    assert len(_images(result)) == 4, "batch must return one overlay plus three pixel images"
    print("analyze_batch ok")


async def _checkCorrupt(session: ClientSession, bad_path: Path) -> None:
    """Verify a corrupted image is rejected before inference."""
    result = await session.call_tool(
        "matting",
        {"image_path": str(bad_path), "query": "person"},
        read_timeout_seconds=600,
    )
    assert result.is_error, "corrupted image must fail"
    assert any("Corrupted" in text or "unreadable" in text for text in _texts(result)), _texts(result)
    print("corrupt image rejected ok")


def _checkArtifacts(workspace: Path) -> None:
    """Verify the expected files landed under the agent cwd."""
    root = workspace / "wevisone_vl_outputs"
    assert len(list(root.rglob("overlay.png"))) == 3
    assert len(list(root.rglob("alpha.png"))) == 3
    assert len(list(root.rglob("cutout.png"))) == 3
    assert len(list(root.rglob("result.json"))) == 3
    print("artifacts ok")


def _checkRequestLog() -> None:
    """Verify request logging captured both calls."""
    records = [
        json.loads(line)
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines()[-3:]
    ]
    tools = [record["tool"] for record in records]
    assert tools == ["matting", "detection", "segmentation"], tools
    assert all(record["prompt"] for record in records)
    print("request log ok")


async def _runChecks(workspace: Path) -> None:
    """Boot the server and run every check against it."""
    before_mib = _gpuUsedMiB()
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        cwd=str(workspace),
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await _checkTools(session)
            await _checkSingle(session)
            await _checkBatch(session)
            bad_path = workspace / "bad.png"
            bad_path.write_bytes(b"not an image")
            await _checkCorrupt(session, bad_path)
            after_mib = _gpuUsedMiB()
    if before_mib is not None and after_mib is not None:
        delta = after_mib - before_mib
        print(f"gpu memory delta: {delta} MiB")
        assert delta < MODEL_VRAM_LIMIT_MIB, f"model VRAM not released: +{delta} MiB"
    _checkArtifacts(workspace)
    _checkRequestLog()


def main() -> None:
    """Run the real server test in a temporary agent cwd."""
    with tempfile.TemporaryDirectory(prefix="wevisone-mcp-") as directory:
        asyncio.run(_runChecks(Path(directory)))
    print("all checks passed")


if __name__ == "__main__":
    main()
