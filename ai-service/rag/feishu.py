"""Feishu document loading for RAG through Feishu OpenAPI."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag.text_processing import denoise_text, normalize_text

logger = logging.getLogger("ai-service.rag.feishu")

FEISHU_API_BASE = "https://open.feishu.cn"
FEISHU_URL_RE = re.compile(r"https?://[^\s<>'\"]*feishu\.cn/[^\s<>'\"]+")
DOC_TOKEN_RE = re.compile(r"/(docx?|bitable|base|sheets?|slides|mindnote)/([A-Za-z0-9_-]{12,})")
WIKI_NODE_RE = re.compile(r"/wiki/([A-Za-z0-9_-]{12,})")
WIKI_SPACE_RE = re.compile(r"(?:/wiki/(?:space|spaces)/|[?&]space_id=)([A-Za-z0-9_-]{6,})")
DOCS_V1_TYPES = {"docx"}
DOWNLOADABLE_TYPES = {"file", "md", "txt", "csv", "markdown"}


@dataclass(frozen=True)
class FeishuLinkInfo:
    raw: str
    kind: str
    token: str
    doc_type: str = "docx"


class FeishuOpenAPIClient:
    """Small Feishu OpenAPI client using tenant_access_token."""

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        base_url: str = FEISHU_API_BASE,
        timeout: float = 30.0,
    ):
        self.app_id = app_id if app_id is not None else settings.feishu_app_id
        self.app_secret = app_secret if app_secret is not None else settings.feishu_app_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._tenant_access_token = ""
        self._token_expires_at = 0.0

    def _get_tenant_access_token(self) -> str:
        if self._tenant_access_token and time.time() < self._token_expires_at - 300:
            return self._tenant_access_token
        if not self.app_id or not self.app_secret:
            raise RuntimeError("Feishu credentials are not configured")

        response = httpx.post(
            f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=self.timeout,
        )
        data = _json_response(response)
        token = data.get("tenant_access_token")
        if not token:
            raise RuntimeError("Feishu token response did not include tenant_access_token")
        self._tenant_access_token = token
        self._token_expires_at = time.time() + int(data.get("expire") or 7200)
        return token

    def request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict:
        token = self._get_tenant_access_token()
        response = httpx.request(
            method.upper(),
            f"{self.base_url}{path}",
            params=params,
            json=json_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            timeout=self.timeout,
        )
        return _json_response(response)


def _json_response(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Feishu API returned non-JSON response: HTTP {response.status_code}") from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"Feishu API request failed: HTTP {response.status_code}")
    code = data.get("code", 0)
    if code != 0:
        raise RuntimeError(f"Feishu API error {code}: {data.get('msg') or data.get('message') or 'unknown'}")
    return data


def _normalize_doc_type(doc_type: str) -> str:
    normalized = (doc_type or "docx").lower()
    if normalized == "sheets":
        return "sheet"
    if normalized == "base":
        return "bitable"
    return normalized


def _html_to_text(html: str) -> str:
    if not html.strip():
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["style", "script"]):
        tag.decompose()
    return soup.get_text("\n")


def inspect_feishu_link(url_or_token: str) -> FeishuLinkInfo:
    value = url_or_token.strip()
    if not value:
        raise ValueError("飞书链接为空")

    doc_match = DOC_TOKEN_RE.search(value)
    if doc_match:
        return FeishuLinkInfo(
            raw=value,
            kind="document",
            token=doc_match.group(2),
            doc_type=_normalize_doc_type(doc_match.group(1)),
        )

    space_match = WIKI_SPACE_RE.search(value)
    if space_match:
        return FeishuLinkInfo(raw=value, kind="wiki_space", token=space_match.group(1), doc_type="wiki")

    wiki_match = WIKI_NODE_RE.search(value)
    if wiki_match:
        return FeishuLinkInfo(raw=value, kind="wiki_node", token=wiki_match.group(1), doc_type="wiki")

    if "://" in value or "feishu.cn" in value:
        raise ValueError(f"无法从 URL 中提取文档 token: {url_or_token}")

    if value.lower().startswith(("wiki", "wik")):
        return FeishuLinkInfo(raw=value, kind="wiki_node", token=value, doc_type="wiki")

    return FeishuLinkInfo(raw=value, kind="document", token=value, doc_type="docx")


def find_feishu_links(text: str) -> list[FeishuLinkInfo]:
    links: list[FeishuLinkInfo] = []
    for match in FEISHU_URL_RE.finditer(text):
        try:
            links.append(inspect_feishu_link(match.group(0)))
        except ValueError:
            continue
    return links


def extract_doc_token(url_or_token: str) -> str:
    return inspect_feishu_link(url_or_token).token


def resolve_wiki_node(
    node_token: str,
    obj_type: str = "wiki",
    client: FeishuOpenAPIClient | None = None,
) -> dict:
    api = client or FeishuOpenAPIClient()
    params = {"token": node_token}
    if obj_type:
        params["obj_type"] = obj_type
    data = api.request("GET", "/open-apis/wiki/v2/spaces/get_node", params=params)
    node = data.get("data", {}).get("node", {})
    if not node.get("obj_token"):
        raise RuntimeError(f"飞书 Wiki 节点未返回实际文档 token: {node_token}")
    return node


def list_wiki_nodes(
    space_id: str,
    parent_node_token: str = "",
    page_size: int = 50,
    page_token: str = "",
    client: FeishuOpenAPIClient | None = None,
) -> dict:
    api = client or FeishuOpenAPIClient()
    params = {"page_size": max(1, min(page_size, 50))}
    if parent_node_token:
        params["parent_node_token"] = parent_node_token
    if page_token:
        params["page_token"] = page_token
    data = api.request(
        "GET",
        f"/open-apis/wiki/v2/spaces/{quote(space_id, safe='')}/nodes",
        params=params,
    )
    inner = data.get("data", {})
    return {
        "items": inner.get("items", []),
        "has_more": bool(inner.get("has_more", False)),
        "page_token": inner.get("page_token", ""),
    }


def walk_wiki_nodes(
    space_id: str,
    max_pages: int = 200,
    client: FeishuOpenAPIClient | None = None,
) -> list[dict]:
    nodes: list[dict] = []
    queue: list[str] = [""]
    pages = 0
    api = client or FeishuOpenAPIClient()

    while queue and pages < max_pages:
        parent = queue.pop(0)
        page_token = ""
        while pages < max_pages:
            pages += 1
            page = list_wiki_nodes(
                space_id,
                parent_node_token=parent,
                page_token=page_token,
                client=api,
            )
            for node in page["items"]:
                nodes.append(node)
                if node.get("has_child") and node.get("node_token"):
                    queue.append(node["node_token"])
            if not page["has_more"]:
                break
            page_token = page["page_token"]

    return nodes


class FeishuDocLoader:
    """Fetch Feishu docs and split them into LangChain documents."""

    def __init__(
        self,
        client: FeishuOpenAPIClient | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.client = client or FeishuOpenAPIClient()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def fetch_content(self, doc_token: str, doc_type: str = "docx") -> dict:
        doc_type = _normalize_doc_type(doc_type)
        if doc_type in DOWNLOADABLE_TYPES:
            raise RuntimeError(f"暂不支持直接同步飞书文件类型：{doc_type}")
        if doc_type in DOCS_V1_TYPES:
            data = self.client.request(
                "GET",
                "/open-apis/docs/v1/content",
                params={
                    "doc_token": doc_token,
                    "doc_type": "docx",
                    "content_type": "markdown",
                    "lang": "zh",
                },
            )
            inner = data.get("data", {})
            return {
                "markdown": inner.get("content") or "",
                "title": inner.get("title") or doc_token,
                "doc_id": doc_token,
                "doc_type": "docx",
            }

        body = {
            "export_option": {
                "export_block_id": False,
                "export_cite_extra_data": False,
                "export_style_attrs": False,
            },
            "format": "xml",
        }
        data = self.client.request(
            "POST",
            f"/open-apis/docs_ai/v1/documents/{quote(doc_token, safe='')}/fetch",
            json_data=body,
        )
        inner = data.get("data", {})
        document = inner.get("document", inner)
        content = document.get("content") or document.get("markdown") or document.get("text") or inner.get("content") or ""
        return {
            "markdown": _html_to_text(content) if "<" in content and ">" in content else content,
            "title": document.get("title") or inner.get("title") or doc_token,
            "doc_id": document.get("document_id") or inner.get("document_id") or doc_token,
            "doc_type": doc_type,
        }

    def fetch_raw_doc(
        self,
        link: FeishuLinkInfo,
        title: str = "",
    ) -> dict:
        """09 票据：拉取单个飞书文档原始内容，返回 raw dict（不切分）。

        canonical ingestion 路径使用：父子切分由 IngestionPipeline.ingest_content 负责。
        与 load_and_split 共享同一解析逻辑（resolve_wiki_node + fetch_content），但不切分。

        Args:
            link: 飞书链接信息（由 inspect_feishu_link 解析得到）
            title: 期望标题（可选，未提供时用 fetched.title 或 token）

        Returns:
            dict 含字段：
                markdown: 规范化后的 Markdown 文本
                title: 文档标题
                doc_id: 文档 ID（feishu_doc_token）
                doc_type: 文档类型（docx/wiki 等）
                source_url: 原始链接
                feishu_doc_token: 飞书文档 token
                feishu_doc_type: 飞书文档类型
                feishu_wiki_node_token: wiki node token（仅 wiki_node 链接）

        Raises:
            ValueError: wiki_space 链接（需走批量同步 sync_from_space）
            RuntimeError: 文档内容为空或 API 请求失败
        """
        if link.kind == "wiki_space":
            raise ValueError(
                "飞书知识库空间链接需要使用 default-sync 或 sync_from_space 批量同步"
            )

        wiki_node_token = ""
        effective_token = link.token
        effective_type = _normalize_doc_type(link.doc_type)
        effective_title = title

        if link.kind == "wiki_node":
            node = resolve_wiki_node(link.token, client=self.client)
            wiki_node_token = node.get("node_token") or link.token
            effective_token = node["obj_token"]
            effective_type = _normalize_doc_type(node.get("obj_type") or "docx")
            effective_title = effective_title or node.get("title") or effective_token

        fetched = self.fetch_content(effective_token, effective_type)
        markdown = denoise_text(normalize_text(fetched.get("markdown") or ""))
        if not markdown:
            raise RuntimeError(f"飞书文档内容为空：{effective_token}")

        final_title = effective_title or fetched.get("title") or effective_token

        raw = {
            "markdown": markdown,
            "title": final_title,
            "doc_id": fetched.get("doc_id") or effective_token,
            "doc_type": effective_type,
            "source_url": link.raw,
            "feishu_doc_token": effective_token,
            "feishu_doc_type": effective_type,
        }
        if wiki_node_token:
            raw["feishu_wiki_node_token"] = wiki_node_token
        return raw

    def load_and_split(self, doc_token: str, title: str = "", doc_type: str = "docx") -> list[Document]:
        link = inspect_feishu_link(doc_token)
        wiki_node_token = ""
        effective_token = link.token
        effective_type = _normalize_doc_type(doc_type or link.doc_type)
        effective_title = title

        if link.kind == "wiki_space":
            raise ValueError("飞书知识库空间链接需要使用 default-sync 或 sync_from_space 批量同步")
        if link.kind == "wiki_node":
            node = resolve_wiki_node(link.token, client=self.client)
            wiki_node_token = node.get("node_token") or link.token
            effective_token = node["obj_token"]
            effective_type = _normalize_doc_type(node.get("obj_type") or "docx")
            effective_title = effective_title or node.get("title") or effective_token
        elif link.doc_type != "docx" and doc_type == "docx":
            effective_type = link.doc_type

        fetched = self.fetch_content(effective_token, effective_type)
        text = denoise_text(normalize_text(fetched.get("markdown") or ""))
        if not text:
            raise RuntimeError(f"飞书文档内容为空：{effective_token}")

        docs = self.splitter.create_documents([text])
        final_title = effective_title or fetched.get("title") or effective_token
        chunk_count = len(docs)
        for index, doc in enumerate(docs):
            content_hash = hashlib.sha1(
                f"feishu:{effective_token}:{index}:{doc.page_content}".encode("utf-8")
            ).hexdigest()
            doc.metadata.update({
                "id": content_hash,
                "source": "feishu",
                "title": final_title,
                "feishu_doc_token": effective_token,
                "feishu_doc_type": effective_type,
                "chunk_index": index,
                "chunk_count": chunk_count,
                "content_hash": content_hash,
            })
            if wiki_node_token:
                doc.metadata["feishu_wiki_node_token"] = wiki_node_token
        return docs


class FeishuDefaultSyncer:
    """Batch-sync Feishu default sources into the active vector store."""

    def __init__(self, loader: FeishuDocLoader, retriever):
        self.loader = loader
        self.retriever = retriever
        self.synced_documents: list[Document] = []

    def sync_from_space(self, space_id: str) -> list[dict]:
        nodes = walk_wiki_nodes(space_id, client=self.loader.client)
        summaries: list[dict] = []
        for node in nodes:
            token = node.get("obj_token") or node.get("node_token") or ""
            if not token:
                continue
            doc_type = _normalize_doc_type(node.get("obj_type") or "docx")
            title = node.get("title") or token
            try:
                docs = self.loader.load_and_split(token, title=title, doc_type=doc_type)
                # 10 票据 legacy：pgvector/memory 后端的直接写入路径，留待 Ticket 14 cutover 移除。
                # SQLite 后端走 iter_raw_docs_from_space + IngestionPipeline.ingest_content，
                # 不会实例化 FeishuDefaultSyncer(retriever=非None)，不会走到此处。
                self.retriever.add_documents(docs, status="feishu")
                self.synced_documents.extend(docs)
                summaries.append({"token": token, "title": title, "chunks": len(docs)})
            except Exception as exc:
                logger.warning("Feishu node sync failed: %s", exc)
                summaries.append({"token": token, "title": title, "chunks": 0, "error": str(exc)})
        return summaries

    def sync_from_folder(self, folder_token: str) -> list[dict]:
        files = search_feishu_docs(folder_token=folder_token, client=self.loader.client).get("results", [])
        summaries: list[dict] = []
        for item in files:
            meta = item.get("result_meta", item)
            token = meta.get("token") or meta.get("file_token") or ""
            if not token:
                continue
            title = meta.get("title") or meta.get("name") or token
            doc_type = _normalize_doc_type(meta.get("doc_type") or meta.get("type") or "docx")
            try:
                docs = self.loader.load_and_split(token, title=title, doc_type=doc_type)
                # 10 票据 legacy：pgvector/memory 后端的直接写入路径，留待 Ticket 14 cutover 移除。
                # SQLite 后端走 iter_raw_docs_from_folder + IngestionPipeline.ingest_content，
                # 不会实例化 FeishuDefaultSyncer(retriever=非None)，不会走到此处。
                self.retriever.add_documents(docs, status="feishu")
                self.synced_documents.extend(docs)
                summaries.append({"token": token, "title": title, "chunks": len(docs)})
            except Exception as exc:
                logger.warning("Feishu folder file sync failed: %s", exc)
                summaries.append({"token": token, "title": title, "chunks": 0, "error": str(exc)})
        return summaries

    def iter_raw_docs_from_space(self, space_id: str) -> tuple[list[dict], list[dict]]:
        """09 票据：从飞书 wiki 空间拉取所有文档 raw 内容（不切分）。

        canonical ingestion 路径使用：父子切分由调用方通过 IngestionPipeline.ingest_content 负责。
        单份失败 try/except 跳过并收集到 failed_docs，不中断其他文档（验收 4）。

        Args:
            space_id: 飞书 wiki space ID

        Returns:
            (raw_docs, failed_docs) 元组：
                raw_docs: 成功拉取的文档列表，每个 dict 含 fetch_raw_doc 返回的字段
                failed_docs: 失败的文档列表，每个 dict 含 token/title/error 字段
        """
        nodes = walk_wiki_nodes(space_id, client=self.loader.client)
        raw_docs: list[dict] = []
        failed_docs: list[dict] = []
        for node in nodes:
            token = node.get("obj_token") or node.get("node_token") or ""
            if not token:
                continue
            doc_type = _normalize_doc_type(node.get("obj_type") or "docx")
            title = node.get("title") or token
            try:
                # 09 票据验收 3：raw 必须是完整 URL，让 fetch_raw_doc 写入真实 source_url（非 token）
                link = FeishuLinkInfo(
                    raw=f"https://feishu.cn/{doc_type}/{token}",
                    kind="document",
                    token=token,
                    doc_type=doc_type,
                )
                raw = self.loader.fetch_raw_doc(link, title=title)
                raw_docs.append(raw)
            except Exception as exc:
                logger.warning("Feishu wiki node fetch failed: %s", exc)
                failed_docs.append({"token": token, "title": title, "error": str(exc)})
        return raw_docs, failed_docs

    def iter_raw_docs_from_folder(self, folder_token: str) -> tuple[list[dict], list[dict]]:
        """09 票据：从飞书文件夹拉取所有文档 raw 内容（不切分）。

        canonical ingestion 路径使用：父子切分由调用方通过 IngestionPipeline.ingest_content 负责。
        单份失败 try/except 跳过并收集到 failed_docs，不中断其他文档（验收 4）。

        Args:
            folder_token: 飞书文件夹 token

        Returns:
            (raw_docs, failed_docs) 元组：
                raw_docs: 成功拉取的文档列表
                failed_docs: 失败的文档列表
        """
        files = search_feishu_docs(
            folder_token=folder_token, client=self.loader.client
        ).get("results", [])
        raw_docs: list[dict] = []
        failed_docs: list[dict] = []
        for item in files:
            meta = item.get("result_meta", item)
            token = meta.get("token") or meta.get("file_token") or ""
            if not token:
                continue
            title = meta.get("title") or meta.get("name") or token
            doc_type = _normalize_doc_type(
                meta.get("doc_type") or meta.get("type") or "docx"
            )
            try:
                # 09 票据验收 3：raw 必须是完整 URL，让 fetch_raw_doc 写入真实 source_url（非 token）
                link = FeishuLinkInfo(
                    raw=f"https://feishu.cn/{doc_type}/{token}",
                    kind="document",
                    token=token,
                    doc_type=doc_type,
                )
                raw = self.loader.fetch_raw_doc(link, title=title)
                raw_docs.append(raw)
            except Exception as exc:
                logger.warning("Feishu folder file fetch failed: %s", exc)
                failed_docs.append({"token": token, "title": title, "error": str(exc)})
        return raw_docs, failed_docs


def search_feishu_docs(
    query: str = "",
    folder_token: str = "",
    page_size: int = 50,
    client: FeishuOpenAPIClient | None = None,
) -> dict:
    api = client or FeishuOpenAPIClient()
    params = {"page_size": max(1, min(page_size, 50))}
    if folder_token:
        params["folder_token"] = folder_token
    if query:
        params["query"] = query
    data = api.request("GET", "/open-apis/drive/v1/files", params=params)
    inner = data.get("data", {})
    items = inner.get("files") or inner.get("items") or []
    return {"results": items}
