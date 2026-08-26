from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    chat_id: str = "default"
    scenario: Optional[str] = None
    current_location: Optional["CurrentLocationPayload"] = None
    model_settings: Optional["RuntimeModelSettings"] = None


class ChatResponse(BaseModel):
    content: str
    chat_id: str = "default"


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


class CurrentLocationPayload(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: Optional[float] = None
    province: str = ""
    city: str = ""
    district: str = ""
    adcode: str = ""
    address: str = ""
    source: str = ""


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


class FeishuSyncRequest(BaseModel):
    """同步飞书文档的请求参数。"""

    doc_token: str
    title: str = ""
    doc_type: str = "docx"


class FeishuDocRef(BaseModel):
    """已同步的单篇飞书文档摘要。"""

    token: str
    title: str
    chunks: int
    error: Optional[str] = None


class FeishuSyncResponse(BaseModel):
    """同步飞书文档的响应结果。"""

    status: str = "success"
    documents: list[FeishuDocRef] = Field(default_factory=list)


class FeishuDefaultSyncRequest(BaseModel):
    """从飞书空间/文件夹批量同步文档的请求参数。"""

    space_id: Optional[str] = None
    folder_token: Optional[str] = None
    query: str = ""


class FeishuDefaultSyncResponse(BaseModel):
    """批量同步飞书文档的响应结果。"""

    status: str = "success"
    synced_count: int = 0
    error_count: int = 0
    documents: list[FeishuDocRef] = Field(default_factory=list)


class SSEEvent(BaseModel):
    type: str  # thought, tool_call, tool_result, text, done, error
    content: str
    metadata: Optional[dict] = None


class ModelsFetchRequest(BaseModel):
    base_url: str
    api_key: str


class ModelsFetchResponse(BaseModel):
    models: list[str]



# ── 工具调用确认 ──────────────────────────────────────────


class ConfirmRequest(BaseModel):
    """用户确认/拒绝工具调用的请求。"""

    confirmation_id: str
    action: str = "confirm"  # "confirm" | "reject"


class ConfirmResponse(BaseModel):
    """确认操作的响应。"""

    status: str  # "confirmed" | "rejected" | "not_found" | "already_resolved"
    confirmation_id: str
    tool_name: Optional[str] = None
    tool_result: Optional[dict] = None


class PendingConfirmationItem(BaseModel):
    """单条待确认工具调用的输出结构。"""

    confirmation_id: str
    tool_name: str
    args: dict
    step: int
    status: str
    created_at: float


class PendingConfirmationsResponse(BaseModel):
    """某次对话所有待确认记录的响应。"""

    chat_id: str
    pending: list[PendingConfirmationItem]
