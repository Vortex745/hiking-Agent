import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import tool
from tavily import TavilyClient

from config import settings

# Tavily client is synchronous; use a thread pool for async compatibility
_tavily_executor = ThreadPoolExecutor(max_workers=2)


def _tavily_search_sync(query: str) -> dict:
    client = TavilyClient(api_key=settings.tavily_api_key)
    return client.search(query=query, search_depth="advanced")


@tool
async def web_search(query: str) -> str:
    """Search the web for current information. Use this when you need to find recent news, facts, or data."""
    if not settings.tavily_api_key:
        return "搜索服务未配置：缺少 Tavily API Key。"

    try:
        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(_tavily_executor, _tavily_search_sync, query),
            timeout=15.0,
        )

        results = response.get("results", [])
        if not results:
            return f"未找到关于「{query}」的网络结果。"

        lines = []
        for idx, result in enumerate(results[:5], start=1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "").strip()
            if content:
                lines.append(f"[{idx}] {title}\nURL: {url}\n摘要: {content}")

        return "\n\n".join(lines) if lines else f"未找到关于「{query}」的有效内容。"
    except asyncio.TimeoutError:
        return "搜索超时，请稍后重试。"
    except Exception as e:
        return f"搜索出错: {str(e)}。请稍后重试。"
