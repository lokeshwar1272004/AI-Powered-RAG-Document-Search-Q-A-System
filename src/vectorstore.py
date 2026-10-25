import os
import pickle
from typing  import List,Any


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


from src.config import settings
from src.embeddings import EmbeddingPipeline
from src.logger import setup_logger
from  src.exceptions import VectorStoreError

logger=setup_logger(__name__)    

class FaissVectorStore:

    def __init__(
        self,
        persist_dir:str=None,
        embedding_model:str=None,
        chunk_size:int=None,
        chunk_overlap:int=None
    ):
        self.persist_dir= persist_dir or settings.persist_dir
        os.makedirs(self.persist_dir,exist_ok=True)

        self.embedding_model=embedding_model or settings.embedding_model
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.index=None
        self.metadata:List[Any]=[]

        try:
            self.model=SentenceTransformer(self.embedding_model)
        except Exception as e:
            raise VectorStoreError(f"fail to load embedding model: {e}") from e

        logger.info(f"FassiVectorstore intialized (persist_dir={persist_dir})")

    def build_from_document(self,document:List[Any]):
        logger.info(f"Building vectorstore from {len(document)} raw document")

        emd_pipe=EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks=emd_pipe.chunk_documents(document)
        embeddings=emd_pipe.embed_chunks(chunks)
        metadatas=[{"text":chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"),metadatas)
        self.save()
        logger.info(f"Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self,embeddings:np.ndarray,metadatas:List[Any]=None):
        if embeddings.size==0:
            raise VectorStoreError("No embedding to add")

        dim=embeddings.shape[1]
        if self.index is None:
            self.index=faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        logger.info(f"Added {embeddings.shape[0]} vector(s) to FAISS index")

    def save(self):
        faiss_path=os.path.join(self.persist_dir,"faiss.index")
        mata_path=os.path.join(self.persist_dir,"metadata.pkl")
        try:
            faiss.write_index(self.index,faiss_path)
            with open(mata_path,"wb") as f:
                pickle.dump(self.metadata,f)
        except Exception as e:
            raise VectorStoreError(f"Failed to save vector store: {e}") from e
        logger.info(f"Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path=os.path.join(self.persist_dir,"faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise VectorStoreError(
                f"No saved vector store found at {self.persist_dir}. "
                "Build one first with build_from_documents()."
            )
        try:
            self.index=faiss.read_index(faiss_path)
            with open(meta_path,"rb") as f:
                self.metadata=pickle.load(f)

        except Exception as e:
            raise VectorStoreError(f"Failed to load vector store: {e}") from e
        logger.info(f"Loaded FAISS index and metadata from {self.persist_dir}")

    def search(self,quey_embedding:np.array,top_k:int=None):
        if self.index is None:
            raise VectorStoreError(
                "Vector store has no index loaded — call load() or build_from_documents() first."
            )

        top_k=top_k or settings.top_k
        dist,indices=self.index.search(quey_embedding,top_k)

        result=[]
        for idx,dist in zip(indices[0],dist[0]):
            if idx==-1:
                continue
            meta=self.metadata[idx] if idx < len(self.metadata) else None
            result.append({"index":int(idx),"distance":float(dist),"metadata":meta})
        return result

    def query(self,query_text:str=None,top_k:int=None):
        logger.info(f"Querying vector store for: {query_text!r}")
        query_emb=self.model.encode([query_text]).astype("float32")
        return self.search(query_emb,top_k=top_k)

if __name__=="__main__":
    from src.data_loader import load_all_documents
    store=FaissVectorStore()
    docs=load_all_documents()
    store.build_from_document(docs)

    results=store.query("text query",top_k=3)
    print("\nTop result")
    for r in results:
        preview = (r["metadata"] or {}).get("text", "")[:100]
        print(f"  distance={r['distance']:.4f} | {preview}")







        