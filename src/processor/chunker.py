from typing import List
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models import Document
from src.processor.base import BaseChunker

class ParentChildChunker(BaseChunker):
    def __init__(self, parent_chunk_size: int = 2000, child_chunk_size: int = 400, child_chunk_overlap: int = 40):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, document: Document) -> List[Document]:
        text = document.content
        
        # 1. Split into Parent Chunks
        parent_chunks = self.parent_splitter.split_text(text)
        
        child_documents = []
        
        for parent_text in parent_chunks:
            parent_id = str(uuid4())
            
            # 2. Split Parent into Child Chunks
            child_chunks = self.child_splitter.split_text(parent_text)
            
            for child_text in child_chunks:
                # Create new metadata with parent info
                new_metadata = document.metadata.model_copy()
                new_metadata.parent_id = parent_id
                new_metadata.parent_text = parent_text
                
                child_documents.append(Document(
                    content=child_text,
                    metadata=new_metadata
                ))
                
        return child_documents
