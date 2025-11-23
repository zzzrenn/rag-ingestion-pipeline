from typing import List, Optional, Literal
from uuid import UUID, uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field

SourceType = Literal['pdf', 'markdown']
ProductType = Literal['product_a', 'product_b', 'product_c', 'general']
ContentType = Literal['faq', 'description', 'price', 'terms', 'other']

class DocMetadata(BaseModel):
    source_type: SourceType
    product: ProductType = 'general'
    content_type: ContentType = 'other'
    created_at: datetime = None # Simplified for debugging
    
    # Conditional Fields
    page_number: Optional[int] = None
    source_filename: Optional[str] = None
    
    # Parent-Child Fields
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

class Document(BaseModel):
    id: UUID = None
    content: str
    metadata: DocMetadata
    embedding: Optional[List[float]] = None
    score: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.id is None:
            self.id = uuid4()
