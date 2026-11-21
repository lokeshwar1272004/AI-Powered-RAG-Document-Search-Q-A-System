import os
from contextlib import asynccontextmanager
from typing import List,Any
import uuid
from pathlib import Path


import numpy as np
from fastapi import HTTPException,FastAPI,UploadFile,File
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader,TextLoader


from src.config import settings
from src.logger import setup_logger
from src.exceptions import SerachError
from src.database import init_db,add_docuement,mark_processed,mark_failed,list_documents
from src.embeddings import EmbeddingPipeline
from src.search import RAGSearch


logger=setup_logger(__name__)

rag_search:RAGSearch=None


@asynccontextmanager
async def lifespan(app:FastAPI):
    global rag_search
    logger.info("Starting up: initializing database and RAG pipeline...")
    init_db()
    rag_search=RAGSearch()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app=FastAPI(title="RAG Document Search API",lifespan=lifespan)


class QueryRequest(BaseModel):
    query:str
    top_k:int=None

class QueryResponse(BaseModel):
    answer:str

class DocumentOut(BaseModel):
    id:int
    filename:str
    status:str
    chunk_count:int

@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/query",response_model=QueryResponse)
def query(request:QueryRequest):
    try:
        answer=rag_search.search_and_summarize(request.query,request.top_k)
        return QueryResponse(answer=answer)
    except SerachError as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/documents",response_model=List[DocumentOut])
def get_documents():
    docs=list_documents()
    return [DocumentOut(id=d.id,filename=d.filename,status=d.status,chunk_count=d.chunk_count) for d in docs]


@app.post("/upload", response_model=DocumentOut)
def update_document(file: UploadFile = File(...)):
    filename = Path(file.filename).name
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".pdf", ".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf and .txt files are supported."
        )

    # Check for duplicate filenames
    existing_documents = list_documents()

    if any(doc.filename == filename for doc in existing_documents):
        raise HTTPException(
            status_code=409,
            detail=f"'{filename}' has already been uploaded."
        )

    os.makedirs(settings.data_dir, exist_ok=True)
    save_path = os.path.join(settings.data_dir, filename)

    with open(save_path, "wb") as f:
        f.write(file.file.read())

    doc_id = add_docuement(filename)

    try:
        loaded=PyPDFLoader(save_path).load() if ext==".pdf" else TextLoader(save_path).load()
        pipeline=EmbeddingPipeline()
        chunks=pipeline.chunk_documents(loaded)
        embeddings=pipeline.embed_chunks(chunks)
        # metadatas=[{"text":c.page_content} for c in chunks]
        metadatas = [
                        {
                            "text": chunk.page_content,
                            "filename": file.filename,
                            "document_id": doc_id,
                        }
                        for chunk in chunks
                    ]

        rag_search.vectorstore.add_embeddings(np.array(embeddings).astype("float32"),metadatas=metadatas)
        rag_search.vectorstore.save()

        mark_processed(doc_id, chunk_count=len(chunks))
        logger.info(f"Uploaded and indexed {file.filename} ({len(chunks)} chunks)")

    except Exception as e:
        mark_failed(doc_id)
        logger.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

    return DocumentOut(id=doc_id, filename=file.filename, status="processed", chunk_count=len(chunks))





