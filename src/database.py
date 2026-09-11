from datetime import datetime
from sqlalchemy import create_engine,Column,Integer,String,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker


from src.config import settings
from src.logger import setup_logger
from src.exceptions import RAGAppError

logger=setup_logger(__name__)
Base=declarative_base()


class DocumentRecord(Base):
    __tablename__="document"

    id=Column(Integer,primary_key=True,autoincrement=True)
    filename=Column(String,nullable=False)
    uploaded_at=Column(DateTime,default=datetime.utcnow)
    chunk_count=Column(Integer,default=0)
    status=Column(String,default="Pending")

engine=create_engine(settings.database_url)
SessionLocal=sessionmaker(bind=engine)


def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified.")
    except Exception as e:
        raise RAGAppError(f"Failed to initialize database: {e}") from e


def add_docuement(filename:str)->int:
    session = SessionLocal()
    try:
        record=DocumentRecord(filename=filename,status="pending")
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(f"Recorded new document: {filename} (id={record.id})")
        return record.id
    
    finally:
        session.close()




def mark_processed(document_id: int, chunk_count: int):
    """Update a document's status to 'processed' with its chunk count."""
    session = SessionLocal()
    try:
        record = session.get(DocumentRecord, document_id)
        if record:
            record.status = "processed"
            record.chunk_count = chunk_count
            session.commit()
            logger.info(f"Document id={document_id} marked processed ({chunk_count} chunks)")
    finally:
        session.close()

def mark_failed(document_id: int):
    """Update a document's status to 'failed'."""
    session = SessionLocal()
    try:
        record = session.get(DocumentRecord, document_id)
        if record:
            record.status = "failed"
            session.commit()
            logger.warning(f"Document id={document_id} marked failed")
    finally:
        session.close()

def list_documents():
    """Return all tracked documents, newest first."""
    session = SessionLocal()
    try:
        return session.query(DocumentRecord).order_by(DocumentRecord.uploaded_at.desc()).all()
    finally:
        session.close()


if __name__=="__main__":
    init_db()
    doc_id = add_docuement("example.pdf")
    mark_processed(doc_id, chunk_count=5)

    print("\nAll tracked documents:")
    for doc in list_documents():
        print(f"  id={doc.id} | {doc.filename} | status={doc.status} | "
              f"chunks={doc.chunk_count} | uploaded_at={doc.uploaded_at}")
