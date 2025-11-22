from typing import List
from src.models import Document
from src.processor.base import BaseChunker

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def chunk(self, document: Document) -> List[Document]:
        text = document.content
        chunks = self._split_text(text, self.separators)
        
        documents = []
        for chunk_text in chunks:
            new_metadata = document.metadata.model_copy()
            documents.append(Document(
                content=chunk_text,
                metadata=new_metadata
            ))
            
        return documents

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
                
        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        for split in splits:
            if not split:
                continue
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if new_separators:
                    good_splits.extend(self._split_text(split, new_separators))
                else:
                    good_splits.append(split)
                    
        return self._merge_splits(good_splits, separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            if total + len(split) + len(separator) > self.chunk_size:
                if current_doc:
                    docs.append(separator.join(current_doc))
                    
                    while total > self.chunk_overlap:
                        total -= len(current_doc[0]) + len(separator)
                        current_doc.pop(0)
                        
            current_doc.append(split)
            total += len(split) + len(separator)
            
        if current_doc:
            docs.append(separator.join(current_doc))
            
        return docs
