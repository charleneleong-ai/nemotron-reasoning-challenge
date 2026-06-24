"""Env-driven runtime settings."""

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True
    )

    HF_TOKEN: SecretStr | None = None
    # Accept either GEMINI_API_KEY or GOOGLE_API_KEY for the CoT-augmentation step.
    GEMINI_API_KEY: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )


settings = Settings()
