from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class ChatRequest(BaseModel):
    query: str
    k: int = 4
