"""Shared pytest fixtures."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import pytest
from faker import Faker

from log_rca.storage import LocalFSBackend


@pytest.fixture
def rng() -> random.Random:
    """Deterministic RNG for every test."""
    return random.Random(0)


@pytest.fixture
def fake() -> Faker:
    """Seeded Faker for every test."""
    f = Faker()
    Faker.seed(0)
    return f


@pytest.fixture
def storage(tmp_path: Path) -> LocalFSBackend:
    return LocalFSBackend(tmp_path)


@pytest.fixture
def fixed_start() -> datetime:
    """A fixed datetime to use as a task start time in tests."""
    return datetime(2026, 1, 15, 10, 30, 0)
