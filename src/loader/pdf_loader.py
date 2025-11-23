from typing import List
import asyncio
import fitz  # PyMuPDF
from src.models import Document, DocMetadata
from src.loader.base import BaseLoader
from src.loader.factory import LoaderFactory

@LoaderFactory.register(".pdf")
class PDFLoader(BaseLoader):
    async def load(self, file_path: str, **metadata_kwargs) -> List[Document]:
        """
        Load a PDF file asynchronously and extract text from each page
        
        Args:
            file_path: Path to the PDF file
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            List of Document objects, one per page
        """
        # PyMuPDF is not async, run in thread pool
        documents = await asyncio.to_thread(self._load_sync, file_path, **metadata_kwargs)
        return documents
    
    def _load_sync(self, file_path: str, **metadata_kwargs) -> List[Document]:
        """Synchronous PDF loading"""
        pdf_document = fitz.open(file_path)
        documents = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Build metadata - merge with provided metadata
            # Ensure source_type defaults to 'pdf' if not provided
            metadata_dict = {
                'source_type': 'pdf',
                'page_number': page_num + 1,
                'source_filename': file_path.split('/')[-1],
                **metadata_kwargs  # User-provided metadata can override defaults
            }
            
            metadata = DocMetadata(**metadata_dict)
            doc = Document(content=text, metadata=metadata)
            documents.append(doc)
            
        pdf_document.close()
        return documents
