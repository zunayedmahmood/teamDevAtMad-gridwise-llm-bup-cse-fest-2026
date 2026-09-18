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
    openai_timeout_seconds: float = 5.0
    openai_max_retries: int = 1

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    optimize_deadline_seconds: float = 25.0


settings = Settings()
