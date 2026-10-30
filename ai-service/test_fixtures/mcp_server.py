"""Small MCP server used by the local transport and approval contract tests."""

import argparse
import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


class Payload(BaseModel):
    text: str
    options: dict[str, str]


parser = argparse.ArgumentParser()
parser.add_argument("--http-port", type=int, default=0)
args = parser.parse_args()
server = FastMCP("contract-test", host="127.0.0.1", port=args.http_port or 8000)


def new_tool() -> str:
    return "new tool"


class ImageQuery(BaseModel):
    keyword: str = "mountain"


FIXTURE_IMAGE_BIG = "https://cdn.fixture.test/images/white-cloud-big.jpg?x=fixture"
FIXTURE_IMAGE_PREVIEW = "https://preview.fixture.test/images/white-cloud.jpg!qt_kuan320"
FIXTURE_IMAGE_ORIGIN = "https://origin.fixture.test/images/white-cloud.jpg?token=fixture-secret"
FIXTURE_IMAGE_DOWNLOAD = "https://origin.fixture.test/download/white-cloud.jpg"


@server.tool()
def search_images(query: ImageQuery) -> str:
    """Return one stable image object with public and protected versions."""
    return json.dumps(
        {
            "ok": True,
            "data": {
                "list": [
                    {
                        "pid": 7001,
                        "title": "白云山",
                        "keyword": query.keyword,
                        "big_url": FIXTURE_IMAGE_BIG,
                        "preview_url": FIXTURE_IMAGE_PREVIEW,
                        "origin_url": FIXTURE_IMAGE_ORIGIN,
                        "download_url": FIXTURE_IMAGE_DOWNLOAD,
                    }
                ]
            },
        },
        ensure_ascii=False,
    )


@server.tool()
def echo(payload: Payload) -> str:
    """Record a nested input and return it with the configured test secret."""
    path = Path(os.environ["MCP_TEST_EFFECT_FILE"])
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload.model_dump(), ensure_ascii=False) + "\n")
    if payload.text == "enable_extra":
        server.tool()(new_tool)
    if payload.text == "crash":
        os._exit(42)
    return f"echo:{payload.text}:{payload.options['mode']}:{os.environ.get('MCP_TEST_SECRET', '')}"


if __name__ == "__main__":
    print(os.environ.get("MCP_TEST_SECRET", ""), file=sys.stderr, flush=True)
    server.run(transport="streamable-http" if args.http_port else "stdio")
