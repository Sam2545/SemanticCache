from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration, sourced entirely from the environment.

    No model or connection assumptions are hardcoded in the core.
    """

    model_config = SettingsConfigDict(env_prefix="SEMCACHE_")

    redis_url: str = "redis://localhost:6379"
    default_threshold: float = 0.8
    default_top_k: int = 5


settings = Settings()
