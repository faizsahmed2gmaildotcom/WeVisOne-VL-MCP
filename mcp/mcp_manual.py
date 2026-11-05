#!/usr/bin/env python
"""Talk to the WeVisOne-VL MCP server over stdio, exactly like an MCP client."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

ROOT = Path("/home/kurumi/PycharmProjects/WeVisOne-VL-MCP")
SERVER = ROOT / "mcp" / "wevisone_vl_mcp.py"


async def call(session: ClientSession, name: str, arguments: dict) -> None:
    """Call one tool and print what the agent would receive."""
    result = await session.call_tool(name, arguments, read_timeout_seconds=1200)
    print(f"\nis_error: {result.is_error}")
    for block in result.content:
        if block.type == "text":
            print(block.text)
        elif block.type == "image":
            print(f"[image block: {block.mime_type}, {len(block.data)} base64 chars]")


async def repl(session: ClientSession) -> None:
    """Prompt for a tool name and a JSON argument object until blank."""
    while True:
        try:
            name = input("\ntool name (blank to quit): ").strip()
        except EOFError:
            return
        if not name:
            return
        raw = input("arguments JSON: ").strip() or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"invalid JSON: {exc}")
            continue
        await call(session, name, arguments)


async def main(args: argparse.Namespace) -> None:
    env = dict(os.environ)
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value

    params = StdioServerParameters(
        command=args.python,
        args=[str(SERVER)],
        cwd=str(Path.cwd()),
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if args.list:
                for tool in (await session.list_tools()).tools:
                    first = (tool.description or "").splitlines()[0]
                    print(f"{tool.name}: {first}")
                return
            if args.tool:
                await call(session, args.tool, json.loads(args.arguments))
                return
            await repl(session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python that runs the server")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--list", action="store_true", help="list tools and exit")
    parser.add_argument("tool", nargs="?", help="tool name for a one-shot call")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON arguments for one-shot")
    asyncio.run(main(parser.parse_args()))
