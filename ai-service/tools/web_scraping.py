import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
async def web_scraping(url: str) -> str:
    """Scrape and extract main content from a web page URL."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Try to get main content
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            content = "\n".join(lines[:200])
            return content[:5000]
    except httpx.HTTPStatusError as e:
        return f"HTTP 错误: {e.response.status_code}"
    except httpx.RequestError as e:
        return f"请求失败: {str(e)}"
    except Exception as e:
        return f"抓取出错: {str(e)}"
