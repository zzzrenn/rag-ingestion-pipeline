import fitz  # PyMuPDF
from typing import List
from src.models import Document, DocMetadata, SourceType
from src.loader.base import BaseLoader

class PDFLoader(BaseLoader):
    def load(self, file_path: str, **kwargs) -> List[Document]:
        documents = []
        doc = fitz.open(file_path)
        
        # Extract metadata from kwargs or use defaults
        product = kwargs.get('product')
        content_type = kwargs.get('content_type')
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue
                
            metadata = DocMetadata(
                source_type='pdf',
                source_filename=file_path.split('/')[-1].split('\\')[-1], # Handle both separators
                page_number=page_num + 1
            )
            
            if product:
                metadata.product = product
            if content_type:
                metadata.content_type = content_type
                
            documents.append(Document(
                content=text,
                metadata=metadata
            ))
            
        return documents
