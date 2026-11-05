# WeVisOne-VL MCP

A stdio MCP server that exposes WeVisOne-VL image perception tools to an agent.
The matching skill is at `skill/wevisone-vl/SKILL.md`.

## What it does

Give the agent a local image path and a query. The server runs the appropriate
WeVisOne-VL task, returns the raw model text, and returns rendered images. The
model is loaded for the call and released before the call returns.

## Installation

### 1. Install the Python environment

The server needs Python 3.10 or newer, the project dependencies, and the MCP
packages. A CUDA-capable GPU is strongly recommended; the 2B model needs about
4.7 GB of VRAM.

```bash
git clone https://github.com/faizsahmed2gmaildotcom/WeVisOne-VL-MCP.git
cd WeVisOne-VL-MCP

conda create -n wevisone-vl-mcp python=3.12 -y
conda activate wevisone-vl-mcp

pip install -e .
pip install -r mcp/requirements.txt
```

Check that torch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, install the CUDA build of torch for your driver before
registering the server. CPU inference works but is very slow.

### 2. Configure the server

```bash
cp .env.example .env
```

Edit `.env` if you need a different model, device or memory budget. On an 8 GB
GPU, use the conservative values under [Known Issues](#cuda-out-of-memory-on-8-gb-gpus)
before the first run. The server loads the repository-root `.env` itself, so
client environment filtering does not matter. Process environment variables
still override the file.

The first call downloads `tencent/WeVisOne-VL-2B` (about 4.6 GB) into the
Hugging Face cache. Later calls reuse it.

### 3. Record absolute paths

Agents start MCP servers without your interactive shell, so do not rely on
`conda activate` or on `python` being on `PATH`. Resolve the two absolute paths
once and use them in every client configuration:

```bash
cd /absolute/path/to/WeVisOne-VL-MCP
export WEVISONE_PYTHON="$(which python)"
export WEVISONE_SERVER="$(pwd)/mcp/wevisone_vl_mcp.py"
echo "$WEVISONE_PYTHON"
echo "$WEVISONE_SERVER"
```

The examples below use `/opt/wevisone-vl-mcp/bin/python` and
`/srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py` as placeholders. Replace them with
your values.

The server writes `wevisone_vl_outputs/` under its own working directory. Most
clients start stdio servers in the project or session directory, which is the
desired agent cwd. If a client lets you configure a working directory (Codex
and OpenCode use `cwd`), set it to the workspace where results should appear,
not to the repository root.

The model is loaded and released on every call. The first call is slow because
of the download and model load. The timeout examples below are deliberately
generous.

### 4. Register with your agent

#### Claude Code

User scope, from a terminal:

```bash
claude mcp add --scope user --transport stdio \
  wevisone-vl -- \
  /opt/wevisone-vl-mcp/bin/python \
  /srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py
```

Or add a project-scoped `.mcp.json` at the repository you want the agent to work
in:

```json
{
  "mcpServers": {
    "wevisone-vl": {
      "type": "stdio",
      "command": "/opt/wevisone-vl-mcp/bin/python",
      "args": ["/srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py"],
      "timeout": 600000
    }
  }
}
```

Check with `claude mcp list`, or run `/mcp` inside Claude Code. The `timeout` is
in milliseconds.

#### Codex

```bash
codex mcp add wevisone-vl -- \
  /opt/wevisone-vl-mcp/bin/python \
  /srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py
```

The CLI creates the entry with Codex's default timeout. Add the timeout keys
below to `~/.codex/config.toml`, or write the whole entry there directly:

```toml
[mcp_servers.wevisone-vl]
command = "/opt/wevisone-vl-mcp/bin/python"
args = ["/srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py"]
startup_timeout_sec = 60
tool_timeout_sec = 600
```

Codex defaults to a 60-second tool timeout, which is too short for a cold model
load. Keep `tool_timeout_sec` at 600 or higher. Verify with `codex mcp list`.

#### OpenCode

Add the server to `opencode.json` at the project root, or to the global
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "wevisone-vl": {
      "type": "local",
      "command": [
        "/opt/wevisone-vl-mcp/bin/python",
        "/srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py"
      ],
      "enabled": true,
      "timeout": 600000
    }
  }
}
```

OpenCode's MCP `timeout` is in milliseconds and applies to tool calls. Use the
`wevisone-vl` tools in a prompt, or ask the agent to use the server by name.

#### OpenClaw

Stop the Gateway before changing its MCP registry, then add the server:

```bash
openclaw gateway stop

openclaw mcp add wevisone-vl \
  --command /opt/wevisone-vl-mcp/bin/python \
  --arg /srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py \
  --cwd /absolute/path/to/your/workspace \
  --timeout 600 \
  --connect-timeout 120

openclaw mcp doctor wevisone-vl --probe
```

Or edit `~/.openclaw/openclaw.json` (JSON5):

```json5
{
  mcp: {
    servers: {
      "wevisone-vl": {
        transport: "stdio",
        enabled: true,
        command: "/opt/wevisone-vl-mcp/bin/python",
        args: ["/srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py"],
        cwd: "/absolute/path/to/your/workspace",
        connectionTimeoutMs: 120000,
        requestTimeoutMs: 600000
      }
    }
  }
}
```

Set `cwd` to the workspace where you want `wevisone_vl_outputs/` written. Start
the Gateway again after saving.

#### Hermes

Add the server under `mcp_servers` in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  wevisone-vl:
    command: "/opt/wevisone-vl-mcp/bin/python"
    args:
      - "/srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py"
    timeout: 600
    connect_timeout: 120
    env:
      WEVISONE_DEVICE: "cuda:0"
```

Alternatively:

```bash
hermes mcp add wevisone-vl \
  --command /opt/wevisone-vl-mcp/bin/python \
  --args /srv/WeVisOne-VL-MCP/mcp/wevisone_vl_mcp.py
```

Restart Hermes, then check with `hermes mcp test wevisone-vl`. Hermes starts the
server in the session workspace by default, so artifacts follow the agent's
current directory. The `env` block is optional because the server reads `.env`;
it is useful when you want a per-agent device override.

### 5. Smoke test

Ask the agent to analyze a local image, for example:

```text
Use the wevisone-vl detection tool on /absolute/path/to/image.jpg
with the query "person".
```

The result should contain the raw model text, an `Artifacts:` path, and an
overlay image. You can also run the real end-to-end test without an agent:

```bash
python mcp/test_server.py
```

That test boots the server over stdio, calls the 2B model on the demo images,
and checks artifacts, request logs and VRAM release.

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
