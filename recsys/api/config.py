from pydantic_settings import BaseSettings
from recsys.aggregates import ModelType, Source
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    MODEL_TYPE: ModelType
    MODEL_NAME: str
    SOURCE: Source
    RATING_THRESHOLD: int
    RAW_BUCKET: Optional[str]
    MODEL_BUCKET: Optional[str]
    DB_NAME: str
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
