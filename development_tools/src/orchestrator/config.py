from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    orch_db_url: str = "sqlite:///./orchestrator.db"
    orch_max_iterations: int = 5
    orch_concurrency: int = 4

    cursor_base_url: str = "https://api.cursor.com"
    cursor_api_key: str | None = None
    cursor_use_mock: bool = True

    github_token: str | None = None
    github_dry_run: bool = True
    github_default_base_branch: str = "dev"

    ntfy_enabled: bool = False
    ntfy_base_url: str = "https://ntfy.sh"
    ntfy_topic: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()  # type: ignore


