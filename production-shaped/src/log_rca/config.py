"""
Typed configuration loaded from ``config/settings.yaml`` with environment
variable overrides.

Environment variables use the prefix ``LOG_RCA_`` and a double underscore
for nested fields, e.g.::

    LOG_RCA_DATAGEN__TOTAL_RUNS=2000
    LOG_RCA_STORAGE__BUCKET_ROOT=/data
    LOG_RCA_LLM__MODEL=claude-sonnet-4-6
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default location of the YAML config relative to the repo.
_DEFAULT_YAML = (
    Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
)


class StorageSettings(BaseModel):
    backend: str = Field("local", description="local | gcs (gcs not yet implemented)")
    bucket_root: Path = Field(Path("../fake_gcs_bucket"))
    logs_prefix: str = Field("airflow-logs")
    truth_file: str = Field("_truth.jsonl")


class DatagenSettings(BaseModel):
    seed: int = 42
    days_back: int = 14
    total_runs: int = 500
    failure_rate: float = Field(0.30, ge=0.0, le=1.0)


class LLMSettings(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_output_tokens: int = 1024


class Settings(BaseSettings):
    """Top-level config object."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_RCA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    storage: StorageSettings = Field(default_factory=StorageSettings)
    datagen: DatagenSettings = Field(default_factory=DatagenSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


def load_settings(yaml_path: Path | None = None) -> Settings:
    """Load settings with priority: env vars > YAML file > class defaults.

    We compute the YAML-only and env-only views and then overlay any env
    value that differs from the class default on top of the YAML view.
    """
    path = yaml_path or _DEFAULT_YAML
    yaml_data: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    yaml_settings = Settings(**yaml_data).model_dump()
    env_settings = Settings().model_dump()          # env + defaults, no YAML
    default_settings = Settings.model_construct().model_dump()

    # Walk nested sections and pull env overrides where env differs from default.
    merged = yaml_settings
    for section, env_section in env_settings.items():
        default_section = default_settings.get(section, {})
        if not isinstance(env_section, dict):
            if env_section != default_section:
                merged[section] = env_section
            continue
        for k, v in env_section.items():
            if v != default_section.get(k):
                merged[section][k] = v

    return Settings(**merged)
