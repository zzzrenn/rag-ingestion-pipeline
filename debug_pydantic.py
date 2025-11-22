try:
    import pydantic
    print(f"Pydantic version: {pydantic.VERSION}")
    from src.models import DocMetadata
    print("Import successful")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
