"""Tests for the typed settings loader."""

from __future__ import annotations

from pathlib import Path

from log_rca.config import Settings, load_settings


def test_defaults_apply_when_no_yaml(tmp_path: Path):
    s = load_settings(tmp_path / "missing.yaml")
    assert s.storage.backend == "local"
    assert s.datagen.total_runs == 500
    assert 0.0 <= s.datagen.failure_rate <= 1.0


def test_yaml_overrides_defaults(tmp_path: Path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "datagen:\n"
        "  total_runs: 9\n"
        "  failure_rate: 0.5\n",
        encoding="utf-8",
    )
    s = load_settings(cfg)
    assert s.datagen.total_runs == 9
    assert s.datagen.failure_rate == 0.5
    # untouched fields keep their defaults
    assert s.storage.backend == "local"


def test_env_var_overrides_yaml(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("datagen:\n  total_runs: 9\n", encoding="utf-8")
    monkeypatch.setenv("LOG_RCA_DATAGEN__TOTAL_RUNS", "42")
    s = load_settings(cfg)
    assert s.datagen.total_runs == 42


def test_settings_is_pydantic_model():
    s = Settings()
    # Should serialise without error
    assert "storage" in s.model_dump()
    assert "datagen" in s.model_dump()
    assert "llm" in s.model_dump()
