from typing import Optional
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    title: str
    category: Optional[str]
    embedding_status: str
    created_at: str

    class Config:
        from_attributes = True
