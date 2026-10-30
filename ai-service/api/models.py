from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelsFetchRequest(BaseModel):
    base_url: str
    api_key: str = ""
    section: Literal["llm", "embedding", "rerank", "decision"] | None = None


class ModelsFetchResponse(BaseModel):
    models: list[str]


class LlmConfigSectionResponse(BaseModel):
    base_url: str
    model: str
    api_key_masked: str
    api_key_set: bool


class LlmConfigEmbeddingSectionResponse(LlmConfigSectionResponse):
    dimensions: int | None = None


class LlmConfigDecisionSectionResponse(LlmConfigSectionResponse):
    mode: str


class LlmConfigResponse(BaseModel):
    llm: LlmConfigSectionResponse
    embedding: LlmConfigEmbeddingSectionResponse
    rerank: LlmConfigSectionResponse
    decision: LlmConfigDecisionSectionResponse
    updated_at: str | None
    index_dimensions: int | None
    sources: dict[str, str] | None = None


class LlmConfigSectionUpdate(BaseModel):
    """PUT 请求的单段配置；api_key 为空/省略表示保持已存值。"""

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class LlmConfigEmbeddingSectionUpdate(LlmConfigSectionUpdate):
    dimensions: int | None = None


class LlmConfigDecisionSectionUpdate(LlmConfigSectionUpdate):
    mode: str | None = None


class LlmConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LlmConfigSectionUpdate | None = None
    embedding: LlmConfigEmbeddingSectionUpdate | None = None
    rerank: LlmConfigSectionUpdate | None = None
    decision: LlmConfigDecisionSectionUpdate | None = None


class LlmConfigUpdateResponse(LlmConfigResponse):
    reindex_recommended: bool


class LlmConfigTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: Literal["llm", "embedding", "rerank", "decision"]
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class LlmConfigTestResponse(BaseModel):
    latency_ms: int
    model_count: int
    models: list[str]
