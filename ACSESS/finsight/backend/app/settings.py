"""Application settings loaded from environment variables."""
import logging
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration from .env and environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # SEC API configuration (required)
    sec_user_agent: str
    """SEC requires User-Agent as 'Full Name email@domain.com'"""

    # LLM configuration
    llm_provider: Literal["anthropic", "openai", "remote"] = "remote"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    """Model ID from selected provider (e.g., claude-opus-5, gpt-4-turbo, mistral-7b)"""

    # API keys (at least one required based on provider)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Remote LLM endpoint (for vLLM, ollama, etc.)
    remote_llm_url: str = "http://localhost:8000/v1"
    """URL of remote LLM service (e.g., http://localhost:8000/v1 for vLLM)"""

    # Embeddings
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Paths
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/finsight.db"

    # API
    backend_url: str = "http://localhost:8000"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Debug
    debug: bool = False
    llm_cache_enabled: bool = True

    @field_validator("sec_user_agent")
    @classmethod
    def validate_sec_user_agent(cls, v: str) -> str:
        """SEC User-Agent must contain a name and email."""
        if not v or len(v) < 10:
            raise ValueError(
                "SEC_USER_AGENT must be 'Full Name email@domain.com' "
                "(SEC requires this to prevent abuse)"
            )
        if "@" not in v or " " not in v:
            raise ValueError(
                "SEC_USER_AGENT must be 'Full Name email@domain.com' "
                "(must contain space and @)"
            )
        return v

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, v: str) -> str:
        if not v:
            raise ValueError("LLM_MODEL must be specified")
        return v

    def __init__(self, **data):
        """Validate API key is set for chosen provider."""
        super().__init__(**data)

        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY not set. "
                "Set ANTHROPIC_API_KEY in .env or environment."
            )
        elif self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY not set. "
                "Set OPENAI_API_KEY in .env or environment."
            )
        elif self.llm_provider == "remote" and not self.remote_llm_url:
            raise ValueError(
                "LLM_PROVIDER=remote but REMOTE_LLM_URL not set. "
                "Set REMOTE_LLM_URL in .env (e.g., http://localhost:8000/v1)"
            )

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings()
