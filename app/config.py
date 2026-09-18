from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-sol"
    openai_fallback_model: str = "gpt-6-astra"
    openai_timeout_seconds: float = 8.0
    openai_max_retries: int = 1

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    optimize_deadline_seconds: float = 25.0

    @field_validator("optimize_deadline_seconds")
    @classmethod
    def validate_deadline(cls, v: float) -> float:
        if not (0 < v <= 28.0):
            raise ValueError("optimize_deadline_seconds must be between 0 and 28.0 seconds")
        return v


settings = Settings()
