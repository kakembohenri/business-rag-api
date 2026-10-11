from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_name: str
    embeddings_model_name: str
    index_name: str
    app_env: str
    pinecone_api_key: str
    langsmith_tracing: bool
    langsmith_endpoint: str
    langsmith_project: str
    langsmith_api_key: str
    openai_api_key: str
    redis_url: str
    canary: str
    cache_ttl_seconds: str
    rate_limit: str
    host_api_port: int
    container_api_port: int
    allowed_origins: str

    model_config = SettingsConfigDict(env_file=".env")

    @property
    def is_production(self) -> bool:
        return self.app_env == "prod"

@lru_cache()
def get_settings() -> Settings:
    return Settings()