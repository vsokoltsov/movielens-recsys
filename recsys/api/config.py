from pydantic_settings import BaseSettings, SettingsConfigDict
from recsys.aggregates import ModelType, Source
from functools import lru_cache

class Settings(BaseSettings):
    MODEL_TYPE: ModelType
    MODEL_NAME: str
    SOURCE: Source
    RATING_THRESHOLD: int

    # model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore