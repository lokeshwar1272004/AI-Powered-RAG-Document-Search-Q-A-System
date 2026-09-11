class RAGAppError(Exception):
    pass
class DocumentLoadError(RAGAppError):
    pass
class EmbeddingError(RAGAppError):
    pass
class VectorStoreError(RAGAppError):
    pass
class SerachError(RAGAppError):
    pass