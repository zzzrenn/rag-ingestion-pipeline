import unicodedata
import re
from src.processor.base import BaseCleaner

class SimpleCleaner(BaseCleaner):
    def clean(self, text: str) -> str:
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
