from src.processor.chunker import ParentChildChunker
from src.models import Document, DocMetadata

def test_chunking():
    print("Testing ParentChildChunker with sentence splitting...")
    
    # Sample text with clear sentence boundaries
    text = (
        # "This is the first sentence. This is the second sentence. "
        # "This is the third sentence?This is the fourth sentence! "
        # "Here is a paragraph break.\n\n"
        # "New paragraph starts here. It has more sentences. "
        "We want to ensure that splits happen at these punctuation marks if possible. Sometimes it is too long."
        # "臺灣（俗字寫作台灣），西方國家在歷史上亦稱福爾摩沙（葡萄牙語：Formosa），是位於東亞、太平洋西北側的島嶼。地處琉球群島與菲律賓群島之間，西隔臺灣海峽與中國大陸相望，海峽距離約130公里，周圍海域從3點鐘方向以順時鐘排序分別為太平洋（菲律賓海）、巴士海峽、南海、臺灣海峽、東海。"
    )
    
    doc = Document(content=text, metadata=DocMetadata(source_type='markdown'))
    
    # Initialize chunker with small size to force splitting
    # Parent chunk size small enough to force splits
    chunker = ParentChildChunker(parent_chunk_size=100, child_chunk_size=50, child_chunk_overlap=0)
    
    chunks = chunker.chunk(doc)
    
    print(f"Original Text Length: {len(text)}")
    print(f"Generated {len(chunks)} child chunks.")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content: '{chunk.content}'")
        print(f"Parent ID: {chunk.metadata.parent_id}")
        
        # Verify no split in middle of sentence (heuristic)
        # Ideally chunks should end with punctuation or be complete phrases
        
if __name__ == "__main__":
    test_chunking()
