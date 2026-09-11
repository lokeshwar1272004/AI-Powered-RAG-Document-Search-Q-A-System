from src.config import settings
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from src.logger import setup_logger
from src.exceptions import SerachError,VectorStoreError

from langchain_groq import ChatGroq

logger=setup_logger(__name__)


class RAGSearch:
    def __init__(self,persist_dir:str=None,embedding_model:str=None,llm_model:str=None):
        self.persist_dir=persist_dir or settings.persist_dir
        self.embedding_model=embedding_model or settings.embedding_model
        self.llm_model=llm_model or settings.llm_model

        self.vectorstore=FaissVectorStore(
            persist_dir=self.persist_dir,
            embedding_model=self.embedding_model
        )
        logger.info(f"Groq LLM initialized: {self.llm_model}")

        try:
            self.vectorstore.load()
            logger.info("Loaded existing vector store from disk.")

        except VectorStoreError:
            logger.info("No existing vector store found — building a new one.")
            documents=load_all_documents(settings.data_dir)
            self.vectorstore.build_from_document(documents)

        if not settings.groq_api_key:
             raise SerachError("GROQ_API_KEY is not set — check your .env file.")

        self.llm=ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=self.llm_model
        )
        logger.info(f"Groq LLM initialized: {self.llm_model}")

    def search_and_summarize(self,query:str=None,top_k:int=None):
        top_k=top_k or settings.top_k
        results=self.vectorstore.query(query,top_k=top_k)
        texts=[r["metadata"].get("text"," ") for r in results if r["metadata"]]
        context="\n\n".join(texts)


        if not context:
            logger.warning(f"No relevant documents found for query: {query!r}")
            return "No relevant documents found for this query."

        prompt=(
            f"Answer the following question using only the context provided.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}"
        )

        try:
            response=self.llm.invoke(prompt)
        except Exception as e:
            raise SerachError(f"llm calling fails:{e}") from e
        return response.content


if __name__ =="__main__":
    rag=RAGSearch()
    answer=rag.search_and_summarize("who is lokeshwar")
    print("\nAnswer:\n",answer)



