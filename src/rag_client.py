import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from src.models import ProductType, ContentType, SourceType
from src.embedder import EmbedderFactory
from src.db import VectorDBFactory

load_dotenv()

class RAGClient:
    def __init__(self):
        embedder_type = os.getenv("EMBEDDER_TYPE", "openai")
        db_type = os.getenv("VECTOR_DB_TYPE", "qdrant")
        
        self.embedder = EmbedderFactory.create(embedder_type)
        self.db = VectorDBFactory.create(db_type)

    def retrieve(self, query: str, filters: Optional[Dict[str, str]] = None, limit: int = 5):
        # 1. Sanitize filters
        sanitized_filters = {}
        if filters:
            for key, value in filters.items():
                if key == 'product':
                    # Simple validation against allowed values
                    if value in ['product_a', 'product_b', 'product_c', 'general']:
                        sanitized_filters[key] = value
                elif key == 'content_type':
                    if value in ['faq', 'description', 'price', 'terms', 'other']:
                        sanitized_filters[key] = value
                elif key == 'source_type':
                    if value in ['pdf', 'markdown']:
                        sanitized_filters[key] = value
                # Ignore unknown filters or invalid values to prevent hallucinated filters
        
        # 2. Embed query
        # We need a dummy document to use the embedder interface, or we can expose a method
        # Since BaseEmbedder takes List[Document], we'll create a dummy one or refactor.
        # Refactoring is better but for now let's just use the internal client of the embedder 
        # or create a dummy document.
        # Actually, let's just use the embedder's client directly if possible or add a method.
        # But BaseEmbedder is abstract. 
        # Let's cheat slightly and use the OpenAI client directly or create a dummy doc.
        # Creating a dummy doc is safer for the interface.
        
        from src.models import Document, DocMetadata
        dummy_doc = Document(content=query, metadata=DocMetadata(source_type='markdown')) # Dummy metadata
        embedded_docs = self.embedder.embed([dummy_doc])
        query_vector = embedded_docs[0].embedding
        
        # 3. Search (Child Chunks)
        child_docs = self.db.search(query_vector, limit=limit * 2, filters=sanitized_filters) # Fetch more children to ensure enough parents
        
        # 4. Deduplicate to Parent Chunks
        seen_parents = set()
        parent_docs = []
        
        for doc in child_docs:
            parent_id = doc.metadata.parent_id
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                
                # Create Parent Document
                # Use parent_text as content
                parent_content = doc.metadata.parent_text or doc.content # Fallback if missing
                
                # Create new metadata without parent fields to avoid confusion? 
                # Or keep them. Let's keep them but update ID.
                new_metadata = doc.metadata.model_copy()
                
                # We construct a new Document representing the Parent
                parent_doc = Document(
                    id=parent_id, # Use parent ID
                    content=parent_content,
                    metadata=new_metadata,
                    score=doc.score # Use max child score? Or just first found.
                )
                parent_docs.append(parent_doc)
                
                if len(parent_docs) >= limit:
                    break
                    
        return parent_docs
