from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    title: str
    category: Optional[str]
    embedding_status: str
    checksum_sha256: Optional[str] = None
    celery_task_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
