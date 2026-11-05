"""Download and cache a WeVisOne-VL model for the MCP server.

Usage:
    python cache_model.py <model-id>

The process exits after the model is cached. The MCP server tracks it through
``.cache.lock``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: cache_model.py <model-id>", file=sys.stderr)
        return 2

    model_id = sys.argv[1]
    model_path = Path(model_id).expanduser()
    if model_path.is_dir():
        print(f"Model already available locally: {model_id}")
        return 0

    print(f"Downloading and caching model: {model_id}", flush=True)
    snapshot_download(model_id)
    print(f"Model cached: {model_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
