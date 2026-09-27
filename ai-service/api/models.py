from pydantic import BaseModel
from typing import Optional


class RAGUploadResponse(BaseModel):
    filename: str
    chunks: int
    status: str = "success"


class RAGDocument(BaseModel):
    id: str
    filename: str
    status: Optional[str] = None
    chunk_count: int


class RuntimeLlmConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class RuntimeEmbeddingConfig(RuntimeLlmConfig):
    dimensions: Optional[int] = None


class RuntimeRerankConfig(RuntimeLlmConfig):
    pass


class RuntimeModelSettings(BaseModel):
    llm: Optional[RuntimeLlmConfig] = None
    embedding: Optional[RuntimeEmbeddingConfig] = None
    rerank: Optional[RuntimeRerankConfig] = None


class RAGQuery(BaseModel):
    question: str
    chat_id: Optional[str] = None
    status: Optional[str] = None
    model_settings: Optional[RuntimeModelSettings] = None
    # 08 票据：澄清双请求 + SSE 契约
    clarification_answers: Optional[list[str]] = None  # resume 时提交澄清答案
    cancel_clarification_id: Optional[str] = None  # 取消进行中的澄清


class ModelsFetchRequest(BaseModel):
    base_url: str
    api_key: str


class ModelsFetchResponse(BaseModel):
    models: list[str]
