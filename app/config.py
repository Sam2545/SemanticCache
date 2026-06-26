from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration, sourced entirely from the environment.

    No model or connection assumptions are hardcoded in the core.
    """

    model_config = SettingsConfigDict(env_prefix="SEMCACHE_")

    backend: str = "memory"  # "memory" | "redis"
    redis_url: str = "redis://localhost:6379"
    default_threshold: float = 0.8
    default_top_k: int = 5
    assumed_llm_ms: float = 800.0
    assumed_llm_cost_usd: float = 0.001
    assumed_tokens_per_call: int = 500


settings = Settings()
