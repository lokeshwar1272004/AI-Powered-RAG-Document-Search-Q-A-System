from pydantic_settings import BaseSettings,SettingsConfigDict



class Settings(BaseSettings):

    #__LLM
    groq_api_key : str
    llm_model :str =  "qwen/qwen3.6-27b"

    #--Embeddings
    embedding_model : str="all-MiniLM-L6-v2"
    chunk_size :int =1000
    chunk_overlap : int=200

    #--storeage/data--
    data_dir : str="data"
    persist_dir:str="faiss_store"

    #--search
    top_k :int=5

    #--api--
    api_host: str="0.0.0.0"
    api_port:int=8000

    #--DataBase
    postgres_host:str="localhost"
    postgres_port:int=5432
    postgres_db:str="rag_doc_search"
    postgres_user:str="postgres"
    postgres_password:str=""

    #logging

    log_level: str = "INFO"
    log_file: str = "logs/app.log"


    @property
    def database_url(self)->str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",
        case_sensitive=False
    )

settings=Settings()
