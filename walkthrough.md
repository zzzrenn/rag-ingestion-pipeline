# Modular RAG Service Walkthrough

## Overview
We have implemented a modular RAG service with the following components:
- **Shared Models**: Pydantic schemas in `src/models.py`.
- **Loaders**: PDF loader using PyMuPDF in `src/loader/`.
- **Processors**: Text cleaner and recursive chunker in `src/processor/`.
- **Embedders**: Factory-based architecture in `src/embedder/`.
  - `EmbedderFactory` allows registering new embedders.
  - `OpenAIEmbedder` registered via decorator.
- **Databases**: Factory-based architecture in `src/db/`.
### Usage
1. **Setup Environment**:
   Fill in `.env` with your API keys.
   ```bash
   pip install -r requirements.txt
   ```

2. **Ingest a File**:
   ```bash
   python ingestion_pipeline.py path/to/doc.pdf --metadata '{"product": "product_a"}'
   ```

3. **Retrieve Documents**:
   ```python
   from src.rag_client import RAGClient
   
   client = RAGClient(db_type='qdrant')
   results = client.retrieve("How does product A work?", filters={"product": "product_a"})
   for doc in results:
       print(doc.content)
   ```
