from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import PyPDFLoader,TextLoader

from src.config import settings
from src.logger import setup_logger
from src.exceptions import DocumentLoadError

logger=setup_logger(__name__)


def load_all_documents(data_dir:str=None)->List[Any]:
    """
    Load all supported files from the data directory and convert them
    into LangChain Document objects.

    Currently supports: PDF, TXT.
    """
    data_dir=data_dir or settings.data_dir
    data_path=Path(data_dir).resolve()
    logger.info(f"Loading Document from {data_path}")

    if not data_path.exists():
        return DocumentLoadError(f"Document directory does not exit:{data_path}")
    documents:List[Any]=[]

    #PDF  files
    pdf_files = list(data_path.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF file(s)")

    for pdf_file in pdf_files:
        try:
            loder=PyPDFLoader(pdf_file)
            loded=loder.load()
            documents.extend(loded)
            logger.info(f"Loaded{len(loded)} page(s) from {pdf_file.name}")
        except Exception as e:
            logger.error(f"Failled loaded pdf{pdf_file}: {e} ")

    #Text Loader
    text_files=list(data_path.glob("**/*.txt"))
    logger.info(f"Found {len(text_files)} text file(s)")
    for text_file in text_files:
        try:
            loder=TextLoader(text_file)
            loded=loder.load()
            documents.extend(loded)
        except Exception as e :
            logger.error(f"failed loaded txt{text_file}: {e}")

    if not documents:
        logger.warning("No document were loded- check your data/folder.")
    logger.info(f"Total document loded{len(documents)}")
    return documents


if __name__=='__main__':
    docs=load_all_documents()
    print(f"\nLoad{len(docs)} document(s) total")
    if docs:
        print("first doc preview:",docs[0].page_content[:200])





