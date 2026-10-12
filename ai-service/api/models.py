from pydantic import BaseModel, Field


class ModelsFetchRequest(BaseModel):
    base_url: str = Field(min_length=1, max_length=2048)
    api_key: str = Field(min_length=1, max_length=4096)


class ModelsFetchResponse(BaseModel):
    models: list[str]
