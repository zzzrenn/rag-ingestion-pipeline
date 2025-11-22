import sys
import os
from unittest.mock import MagicMock, patch

# Mock external dependencies before importing modules that use them
sys.modules['fitz'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['qdrant_client'] = MagicMock()
sys.modules['azure.core.credentials'] = MagicMock()
sys.modules['azure.search.documents'] = MagicMock()

# Now import our code
from src.models import DocMetadata, Document, SourceType
from src.loader.pdf_loader import PDFLoader
from src.processor.cleaner import SimpleCleaner
from src.processor.chunker import RecursiveChunker
from src.embedder.openai_embedder import OpenAIEmbedder
from src.db.qdrant_adapter import QdrantAdapter
from src.rag_client import RAGClient

def test_pipeline_logic():
    print("Testing Pipeline Logic...")
    
    # 1. Test Loader Logic (Mocking fitz)
    print("1. Testing PDFLoader...")
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page 1 content. " * 10
    mock_doc.__iter__.return_value = [mock_page]
    
    with patch('fitz.open', return_value=mock_doc):
        loader = PDFLoader()
        docs = loader.load("dummy.pdf", product="product_a")
        assert len(docs) == 1
        assert docs[0].metadata.product == "product_a"
        assert docs[0].metadata.source_type == 'pdf'
        print("   PDFLoader passed.")

    # 2. Test Processor Logic
    print("2. Testing Processors...")
    cleaner = SimpleCleaner()
    chunker = RecursiveChunker(chunk_size=50)
    
    raw_text = "Hello   World! \x00"
    cleaned = cleaner.clean(raw_text)
    assert cleaned == "Hello World!"
    print("   SimpleCleaner passed.")
    
    doc = Document(content="A" * 100, metadata=docs[0].metadata)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    print("   RecursiveChunker passed.")

    # 3. Test Embedder Logic (Mocking OpenAI)
    print("3. Testing OpenAIEmbedder...")
    with patch('src.embedder.openai_embedder.OpenAI') as mock_openai:
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1]*1536) for _ in chunks]
        mock_client.embeddings.create.return_value = mock_response
        
        embedder = OpenAIEmbedder()
        embedded_docs = embedder.embed(chunks)
        assert embedded_docs[0].embedding is not None
        print("   OpenAIEmbedder passed.")

    # 4. Test DB Logic (Mocking Qdrant)
    print("4. Testing QdrantAdapter...")
    with patch('src.db.qdrant_adapter.QdrantClient') as mock_qdrant:
        db = QdrantAdapter()
        db.upsert(embedded_docs)
        mock_qdrant.return_value.upsert.assert_called()
        print("   QdrantAdapter upsert passed.")
        
        # Test Search
        mock_hit = MagicMock()
        mock_hit.id = "uuid"
        mock_hit.payload = {
            "source_type": "pdf",
            "product": "product_a",
            "content_type": "other",
            "content": "Found content"
        }
        mock_hit.score = 0.9
        mock_qdrant.return_value.query_points.return_value.points = [mock_hit]
        
        results = db.search([0.1]*1536)
        assert len(results) == 1
        assert results[0].content == "Found content"
        print("   QdrantAdapter search passed.")

    # 5. Test RAG Client with Factories
    print("5. Testing RAGClient & Factories...")
    # We need to mock the factory creation or the classes themselves
    
    with patch('src.embedder.openai_embedder.OpenAI') as mock_openai, \
         patch('src.db.qdrant_adapter.QdrantClient') as mock_qdrant:
         
        # Mock Embedder behavior
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1]*1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Mock DB behavior
        mock_qdrant_instance = mock_qdrant.return_value
        mock_hit = MagicMock()
        mock_hit.id = "uuid"
        mock_hit.payload = {
            "source_type": "pdf",
            "product": "product_a",
            "content_type": "other",
            "content": "Found content"
        }
        mock_hit.score = 0.9
        mock_qdrant_instance.query_points.return_value.points = [mock_hit]

        # Set env vars for factory
        os.environ["EMBEDDER_TYPE"] = "openai"
        os.environ["VECTOR_DB_TYPE"] = "qdrant"
        
        client = RAGClient() # Should use factories now
        results = client.retrieve("query", filters={"product": "product_a"})
        
        assert len(results) == 1
        mock_qdrant_instance.query_points.assert_called()
        print("   RAGClient with Factories passed.")

    print("\nAll verification tests passed!")

if __name__ == "__main__":
    test_pipeline_logic()
