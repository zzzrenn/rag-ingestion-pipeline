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
            separators=[
                "\n\n", 
                "\n",
                r"[.?!。？！]+\s*",  # Match punctuation + optional whitespace (for Chinese)
                # r"[,，、]+\s*", # Match comma + optional whitespace (for Chinese)
                # " "
            ],
            is_separator_regex=True,
            keep_separator="start"  # Punctuation+space goes to START of next chunk, we'll strip it later
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=[
                "\n\n", 
                "\n", 
                r"[.?!。？！]+\s*",  # Match punctuation + optional whitespace (for Chinese)
                # r"[,，、]+\s*", # Match comma + optional whitespace (for Chinese)
                # " "
            ],
            is_separator_regex=True,
            keep_separator="start"  # Punctuation+space goes to START of next chunk, we'll strip it later
        )
    
    def _post_process_chunks(self, chunks):
        """Move punctuation from start of chunk to end of previous chunk"""
        processed = []
        for i, chunk in enumerate(chunks):
            if i > 0 and chunk and chunk[0] in ',.?!，、。？！':
                # Find where punctuation ends
                punct_end = 0
                for j, char in enumerate(chunk):
                    if char in ',.?!，、。？！':
                        punct_end = j + 1
                    else:
                        break
                # Move punctuation to previous chunk
                if processed:
                    processed[-1] += chunk[:punct_end]
                chunk = chunk[punct_end:].lstrip()
            if chunk:  # Only add non-empty chunks
                processed.append(chunk)
        return processed

    def chunk(self, document: Document) -> List[Document]:
        text = document.content
        
        # 1. Split into Parent Chunks
        parent_chunks_raw = self.parent_splitter.split_text(text)
        parent_chunks = self._post_process_chunks(parent_chunks_raw)
        
        child_documents = []
        
        for parent_text in parent_chunks:
            parent_id = str(uuid4())
            
            # 2. Split Parent into Child Chunks
            child_chunks_raw = self.child_splitter.split_text(parent_text)
            child_chunks = self._post_process_chunks(child_chunks_raw)
            
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
