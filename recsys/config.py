import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from recsys.aggregates import ModelType, Source, Setup
from functools import lru_cache
from typing import Literal, Union, Annotated
from pydantic import Field, TypeAdapter


class DefaultSettings(BaseSettings):
    MODEL_TYPE: ModelType
    MODEL_NAME: str
    SOURCE: str
    RATING_THRESHOLD: int
    SETUP: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SourceDB(DefaultSettings):
    SOURCE: Literal["db"]
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


class SourceCSV(BaseModel):
    SOURCE: Literal["csv"]
    MOVIELENS_PATH: str


class SetupLocal(DefaultSettings):
    SETUP: Literal["local"]
    MODELS_PATH: str


class SetupCloud(DefaultSettings):
    SETUP: Literal["cloud"]
    RAW_BUCKET: str
    MODEL_BUCKET: str


SourceCfg = Annotated[Union[SourceDB, SourceCSV], Field(discriminator="SOURCE")]
SetupCfg = Annotated[Union[SetupLocal, SetupCloud], Field(discriminator="SETUP")]

_source_adapter: TypeAdapter[SourceCfg] = TypeAdapter(SourceCfg)
_setup_adapter: TypeAdapter[SetupCfg] = TypeAdapter(SetupCfg)


class Settings(BaseModel):
    source: SourceCfg
    setup: SetupCfg

    @property
    def SOURCE(self) -> Source:
        return Source(self.source.SOURCE)

    @property
    def SETUP(self) -> Setup:
        return Setup(self.setup.SETUP)

    @property
    def DATABASE_URL(self) -> str:
        if isinstance(self.source, SourceDB):
            return self.source.DATABASE_URL
        raise AttributeError("DATABASE_URL only when SOURCE=db")

    @property
    def MOVIELENS_PATH(self) -> str:
        if isinstance(self.source, SourceCSV):
            return self.source.MOVIELENS_PATH
        raise AttributeError("MOVIELENS_PATH only when SOURCE=csv")

    @property
    def MODEL_TYPE(self) -> ModelType:
        return self.setup.MODEL_TYPE

    @property
    def MODEL_NAME(self) -> str:
        return self.setup.MODEL_NAME

    @property
    def RATING_THRESHOLD(self) -> int:
        return self.setup.RATING_THRESHOLD

    @property
    def MODELS_PATH(self) -> str:
        if isinstance(self.setup, SetupLocal):
            return self.setup.MODELS_PATH
        raise AttributeError("MODELS_PATH only when SETUP=local")

    @property
    def MODEL_BUCKET(self) -> str:
        if isinstance(self.setup, SetupCloud):
            return self.setup.MODEL_BUCKET
        raise AttributeError("MODEL_BUCKET only when SETUP=cloud")

    @property
    def RAW_BUCKET(self) -> str:
        if isinstance(self.setup, SetupCloud):
            return self.setup.RAW_BUCKET
        raise AttributeError("RAW_BUCKET only when SETUP=cloud")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = dict(os.environ)
    source = _source_adapter.validate_python(env)
    setup = _setup_adapter.validate_python(env)
    return Settings(source=source, setup=setup)
