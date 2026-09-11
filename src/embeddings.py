from typing import List,Any
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


from src.config import settings
from src.exceptions import EmbeddingError
from src.logger import setup_logger

logger=setup_logger(__name__)


class EmbeddingPipeline:
    def __init__(
            self,
            model_name:str=None,
            chunk_overlap:int=None,
            chunk_size:int=None
    ):
        self.model_name=model_name or settings.embedding_model
        self.chunk_size=chunk_size or settings.chunk_size
        self.chunk_overlap=chunk_overlap or settings.chunk_overlap

        try:
            self.model=SentenceTransformer(self.model_name)
        except Exception as e:
            raise EmbeddingError(
                 f"Failed to load embedding model '{self.model_name}': {e}"
            ) from e

        logger.info(f"Loaded Embedding model '{self.model}' ")


    def chunk_documents(self,documents:List[Any])->List[Any]:
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n",' ',"\n",""]

        )

        chunks=splitter.split_documents(documents)
        logger.info(f"Split{len(documents)} document(s) into{len(chunks)} chunks")
        return chunks


    def embed_chunks(self,chunks:List[Any])->np.ndarray:
        if not chunks:
            raise EmbeddingError("No chunk provide for embed")

        texts=[chunk.page_content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunk(s)...")

        try:
            embeddings= self.model.encode(texts,show_progress_bar=True)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings:{e}") from e

        logger.info(f"Generate embeddings with shape{embeddings.shape}")
        return embeddings

if __name__=="__main__":

    from src.data_loader import load_all_documents

    docs=load_all_documents()
    pipeline=EmbeddingPipeline()
    chunks=pipeline.chunk_documents(docs)
    embeddings=pipeline.embed_chunks(chunks)

    print(f"\nGenerated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
                
        
        