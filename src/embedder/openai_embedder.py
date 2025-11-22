import os
from typing import List
from openai import OpenAI
from src.models import Document
from src.embedder.base import BaseEmbedder
from src.embedder.factory import EmbedderFactory

@EmbedderFactory.register("openai")
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def embed(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
            
        texts = [doc.content for doc in documents]
        
        # OpenAI has a limit on batch size, but for simplicity we assume it fits
        # or we could implement batching here.
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        for doc, embedding_data in zip(documents, response.data):
            doc.embedding = embedding_data.embedding
            
        return documents
