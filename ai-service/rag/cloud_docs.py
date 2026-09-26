"""云知识库文档加载模块（基于远程 API）。

从远程 API 拉取文档原始内容（raw dict 列表），分块由 IngestionPipeline 负责。

08 票据：新增 `load_raw_docs` 方法返回 raw dict 列表（不切分），
供 canonical ingestion 路径使用（父子切分由 IngestionPipeline 负责）。
"""

import logging

import httpx

logger = logging.getLogger("ai-service.rag.cloud_docs")


class CloudDocsLoader:
    """云知识库文档加载器：从远程 API 拉取文档原始内容（不切分）。"""

    REQUEST_TIMEOUT = 30.0


    def fetch_all_docs(self, api_url: str) -> list[dict]:
        try:
            resp = httpx.get(api_url, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"云知识库 API 请求失败: {e}")

        docs = data.get("docs", [])
        if not isinstance(docs, list):
            raise RuntimeError(f"云知识库 API 返回格式异常: 期望 docs 为列表，实际为 {type(docs).__name__}")

        return docs

    def load_raw_docs(self, api_url: str) -> list[dict]:
        """08 票据：拉取远端文档列表，返回 raw dict（不切分）。

        canonical ingestion 路径使用：父子切分由 IngestionPipeline.ingest_content 负责。
        每个 dict 含 name/content/size/modifiedAt 字段。

        空内容文档在此层过滤（不返回），避免 canonical ingestion 抛 IngestionError 中断流程。
        """
        raw_docs = self.fetch_all_docs(api_url)
        valid: list[dict] = []
        for raw in raw_docs:
            name = raw.get("name", "unknown")
            content = raw.get("content", "")
            if not content.strip():
                logger.warning("云知识库文档 %s 内容为空，跳过", name)
                continue
            valid.append(raw)
        return valid

