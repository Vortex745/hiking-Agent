"""
MCP Server: Image Search

Provides image search capability via Pexels API.
Supports both stdio and SSE transport modes.

Usage:
  python server.py --transport stdio
  python server.py --transport sse --port 8100
"""

import json
import os
import sys
import argparse

import httpx
from mcp.server.fastmcp import FastMCP

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

mcp = FastMCP("image-search", description="Search for images using Pexels API")


@mcp.tool()
async def search_images(query: str, count: int = 5) -> str:
    """Search for images using the Pexels API.

    Args:
        query: Search keywords for the image (e.g., 'sunset', 'mountain', 'city')
        count: Number of image results to return (max 10, default 5)

    Returns:
        JSON string with image URLs and metadata
    """
    if not PEXELS_API_KEY:
        return json.dumps({
            "error": "PEXELS_API_KEY not configured. Set the PEXELS_API_KEY environment variable.",
            "results": []
        }, ensure_ascii=False)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": PEXELS_API_KEY},
                params={"query": query, "per_page": min(count, 10)},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for photo in data.get("photos", []):
                results.append({
                    "id": photo.get("id"),
                    "url": photo.get("url"),
                    "photographer": photo.get("photographer"),
                    "src": {
                        "original": photo.get("src", {}).get("original"),
                        "large": photo.get("src", {}).get("large"),
                        "medium": photo.get("src", {}).get("medium"),
                        "small": photo.get("src", {}).get("small"),
                    },
                    "alt": photo.get("alt", ""),
                })

            return json.dumps({"query": query, "count": len(results), "results": results}, ensure_ascii=False)
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"Pexels API error: {e.response.status_code}", "results": []})
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Search MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
