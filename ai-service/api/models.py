from pydantic import BaseModel


class ModelsFetchRequest(BaseModel):
    base_url: str
    api_key: str


class ModelsFetchResponse(BaseModel):
    models: list[str]
