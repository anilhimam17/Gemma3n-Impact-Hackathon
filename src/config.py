from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """This class automates the configuration of the Project from its Environment Variables."""
    # Model Configurations
    llm_model_name: str
    embedding_model_name: str

    # Vector Store Paths
    vector_store_path: Path
    asset_path: Path

    # Vector Store Configurations
    top_k: int

    class Config:
        """This class provides access to the environments variables for configuration."""
        env_file: str = ".env"
        env_file_encoding: str = "utf-8"


settings = Settings()  # type: ignore