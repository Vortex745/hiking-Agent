"""Minimal stdio MCP server for the logging coverage test."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stub")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back."""
    return f"echo:{text}"


if __name__ == "__main__":
    mcp.run()
